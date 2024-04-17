module DummyTestModels

using MLJBase
using Distributions

## THE CONSTANT DETERMINISTIC REGRESSOR (FOR TESTING)
##

struct DeterministicConstantRegressor <: MLJBase.Deterministic end

function MLJBase.fit(::DeterministicConstantRegressor, verbosity::Int, X, y)
    fitresult = mean(y)
    cache     = nothing
    report    = nothing
    return fitresult, cache, report
end

MLJBase.reformat(::DeterministicConstantRegressor, X) = (MLJBase.matrix(X),)
MLJBase.reformat(::DeterministicConstantRegressor, X, y) = (MLJBase.matrix(X), y)
MLJBase.selectrows(::DeterministicConstantRegressor, I, A) = (view(A, I, :),)
function MLJBase.selectrows(::DeterministicConstantRegressor, I, A, y)
    return (view(A, I, :), y[I])
end

function MLJBase.predict(::DeterministicConstantRegressor, fitresult, Xnew)
    return fill(fitresult, nrows(Xnew))
end

## THE EphemeralClassifier (FOR TESTING)
## Define a Deterministic Classifier with non-persistent `fitresult`, but which addresses
## this by overloading `save`/`restore`:
struct EphemeralClassifier <: MLJBase.Deterministic end
thing = []

function MLJBase.fit(::EphemeralClassifier, verbosity, X, y)
    # if I serialize/deserialized `thing` then `id` below changes:
    id = objectid(thing)
    p = Distributions.fit(UnivariateFinite, y)
    fitresult = (thing, id, p)
    report  = (features = MLJBase.schema(X).names,)
    return fitresult, nothing, report
end

function MLJBase.predict(::EphemeralClassifier, fitresult, X)
    thing, id, p = fitresult
    id == objectid(thing) ||  throw(ErrorException("dead fitresult"))
    return [mode(p) for _ in 1:MLJBase.nrows(X)]
end

function MLJBase.feature_importances(model::EphemeralClassifier, fitresult, report)
    return [ftr => 1.0 for ftr in report.features]
end

MLJBase.target_scitype(::Type{<:EphemeralClassifier}) = AbstractVector{OrderedFactor{2}}
MLJBase.reports_feature_importances(::Type{<:EphemeralClassifier}) = true

function MLJBase.save(::EphemeralClassifier, fitresult)
    thing, _, p = fitresult
    return (thing, p)
end
function MLJBase.restore(::EphemeralClassifier, serialized_fitresult)
    thing, p = serialized_fitresult
    id = objectid(thing)
    return (thing, id, p)
end

end