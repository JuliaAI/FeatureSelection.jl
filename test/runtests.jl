using FeatureSelection
using Test
using MLJBase
import Distributions
using StableRNGs # for RNGs stable across all julia versions
rng = StableRNGs.StableRNG(123)

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
    

end
