using FeatureSelection, MLJBase, MLJDecisionTreeInterface, StableRNGs, Test
import Distributions

const rng = StableRNG(123)

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
    
#### FEATURE SELECTOR ####

@testset "Feat Selector" begin
    N = 100
    X = (
        Zn   = rand(N),
        Crim = rand(N),
        x3   = categorical(rand("YN", N)),
        x4   = categorical(rand("YN", N))
    )

    # Test feature selection with `features=Symbol[]`
    namesX   = MLJBase.schema(X).names |> collect
    selector = FeatureSelector()
    f,       = MLJBase.fit(selector, 1, X)
    @test f == namesX
    Xt = MLJBase.transform(selector, f, MLJBase.selectrows(X, 1:2))
    @test Set(MLJBase.schema(Xt).names) == Set(namesX)
    @test length(Xt.Zn) == 2

    # Test on selecting features if `features` keyword is defined
    selector = FeatureSelector(features=[:Zn, :Crim])
    f,       = MLJBase.fit(selector, 1, X)
    @test MLJBase.transform(selector, f, MLJBase.selectrows(X, 1:2)) ==
            MLJBase.select(X, 1:2, [:Zn, :Crim])

    # test on ignoring a feature, even if it's listed in the `features`
    selector.ignore = true
    f,   = MLJBase.fit(selector, 1, X)
    Xnew = MLJBase.transform(selector, f, X)
    @test MLJBase.transform(selector, f, MLJBase.selectrows(X, 1:2)) ==
         MLJBase.select(X, 1:2, [:x3, :x4])

    # test error about features selected or excluded in fit.
    selector = FeatureSelector(features=[:x1, :mickey_mouse])
    @test_throws(
        ArgumentError,
        MLJBase.fit(selector, 1, X)
    )
    selector.ignore = true
    @test_logs(
        (:warn, r"Excluding non-existent"),
        MLJBase.fit(selector, 1, X)
    )

    # features must be specified if ignore=true
    @test_throws ArgumentError FeatureSelector(ignore=true)

    # test logs for no features selected when using Bool-Callable function interface:
    selector = FeatureSelector(features= x-> x == (:x1))
   @test_throws(
        ArgumentError,
        MLJBase.fit(selector, 1, X)
    )
    selector.ignore = true
    selector.features = x-> x in [:Zn, :Crim, :x3, :x4]
     @test_throws(
        ArgumentError,
        MLJBase.fit(selector, 1, X)
    )

    # Test model Metadata
    @test MLJBase.input_scitype(selector) == MLJBase.Table
    @test MLJBase.output_scitype(selector) == MLJBase.Table
end

#  To be added with FeatureSelectorRule X = (n1=["a", "b", "a"], n2=["g", "g", "g"], n3=[7, 8, 9],
#               n4 =UInt8[3,5,10],  o1=[4.5, 3.6, 4.0], )
# MLJBase.schema(X)
# Xc = coerce(X,  :n1=>Multiclass, :n2=>Multiclass)

@testset "RecursiveFeatureElimination" begin
    #@test_throws ArgumentError RecursiveFeatureElimination(model = rf)
    # Data For use in testset
    X = rand(rng, 50, 10)
    y = @views(10 .* sin.(pi .* X[:, 1] .* X[:, 2]) + 20 .* (X[:, 3] .- 0.5).^ 2 .+ 10 .* X[:, 4] .+ 5 * X[:, 5])
    Xt = MLJBase.table(X)
    Xnew = MLJBase.table(rand(rng, 50, 10))

    # Constructor
    @test_throws FeatureSelection.ERR_SPECIFY_MODEL RecursiveFeatureElimination()
    reg = DeterministicConstantRegressor()
    @test_throws FeatureSelection.ERR_FEATURE_IMPORTANCE_SUPPORT RecursiveFeatureElimination(model = DeterministicConstantRegressor())
    rf = RandomForestRegressor()
    selector = RecursiveFeatureElimination(model = rf)
    @test selector isa FeatureSelection.DeterministicRecursiveFeatureElimination

    # Fit
    selector_mach = machine(selector, Xt, y)
    fit!(selector_mach)
    selector_fp = fitted_params(selector_mach)
    @test propertynames(selector_fp) == (:features_left, :model_fitresult)
    @test selector_fp.features_left == [:x1, :x2, :x3, :x4, :x5]
    @test selector_fp.model_fitresult == MLJBase.fitted_params(
        selector_mach.model.model, selector_mach.fitresult.model_fitresult
    )
    @test feature_importances(selector_mach) == [
        :x1 => 1.0, :x2 => 1.0, :x3 => 1.0, :x4 => 1.0, :x5 => 1.0, 
        :x6 => 2.0, :x7 => 3.0, :x8 => 4.0, :x9 => 5.0, :x10 => 6.0
    ]

    # predict
    yhat = predict(selector_mach, Xnew)
    @test scitype(yhat) === AbstractVector{Continuous}

    # transform
    trf = transform(selector_mach, Xnew)
    sch = MLJBase.schema(trf)
    @test sch.names === (:x1, :x2, :x3, :x4, :x5)
    @test sch.scitypes === (Continuous, Continuous, Continuous, Continuous, Continuous)
end