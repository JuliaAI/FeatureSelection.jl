using .DummyTestModels
const DTM = DummyTestModels

@testset "RecursiveFeatureElimination" begin
    #@test_throws ArgumentError RecursiveFeatureElimination(model = rf)
    # Data For use in testset
    X = rand(rng, 50, 10)
    y = @views(
        10 .* sin.(
            pi .* X[:, 1] .* X[:, 2]
        ) + 20 .* (X[:, 3] .- 0.5).^ 2 .+ 10 .* X[:, 4] .+ 5 * X[:, 5]
    )
    Xt = MLJBase.table(X)
    Xnew = MLJBase.table(rand(rng, 50, 10))
    Xnew2 = MLJBase.table(rand(rng, 50, 10), names = [Symbol("y$i") for i in 1:10])

    # Constructor
    @test_throws FeatureSelection.ERR_SPECIFY_MODEL RecursiveFeatureElimination()
    reg = DTM.DeterministicConstantRegressor()
    @test_throws(
        FeatureSelection.ERR_FEATURE_IMPORTANCE_SUPPORT, 
        RecursiveFeatureElimination(model = DTM.DeterministicConstantRegressor())
    )
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
    rpt = report(selector_mach)
    @test rpt.ranking == [
        1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    ]

    # predict
    yhat = predict(selector_mach, Xnew)
    @test scitype(yhat) === AbstractVector{Continuous}

    # transform
    trf = transform(selector_mach, Xnew)
    sch = MLJBase.schema(trf)
    @test sch.names === (:x1, :x2, :x3, :x4, :x5)
    @test sch.scitypes === (Continuous, Continuous, Continuous, Continuous, Continuous)
    @test_throws FeatureSelection.ERR_FEATURES_SEEN transform(selector_mach, Xnew2)
end