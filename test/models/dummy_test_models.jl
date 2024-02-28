module DummyTestModels

using MLJBase

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
MLJBase.selectrows(::DeterministicConstantRegressor, I, A, y) =
    (view(A, I, :), y[I])

MLJBase.predict(::DeterministicConstantRegressor, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))
end