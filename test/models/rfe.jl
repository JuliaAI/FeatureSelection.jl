using .DummyTestModels
const DTM = DummyTestModels

@testset "RecursiveFeatureElimination" begin
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
    rf = MLJDecisionTreeInterface.RandomForestRegressor(rng = rng)
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
        :x1 => 6.0, :x2 => 5.0, :x3 => 4.0, :x4 => 3.0, :x5 => 2.0, 
        :x6 => 1.0, :x7 => 1.0, :x8 => 1.0, :x9 => 1.0, :x10 => 1.0
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

@testset "Compare results for RFE with scikit-learn" begin
    ## Sklearn rfe and rfecv
    sklearn = MLJScikitLearnInterface.pyimport("sklearn")
    make_friedman1 = MLJScikitLearnInterface.pyimport("sklearn.datasets"=>"make_friedman1")
    RFE_sklearn = MLJScikitLearnInterface.pyimport("sklearn.feature_selection"=>"RFE")
    RFECV_sklearn = MLJScikitLearnInterface.pyimport("sklearn.feature_selection"=>"RFECV")
    SVR_sklearn = MLJScikitLearnInterface.pyimport("sklearn.svm"=>"SVR")
    sklearn_svr_estimator = SVR_sklearn(kernel="linear")
    sklearn_rfe_selector = RFE_sklearn(
        sklearn_svr_estimator, n_features_to_select=5, step=1
    )
    sklearn_rfecv_selector = RFECV_sklearn(
        sklearn_svr_estimator, step=1, cv=5
    )
    Xs, ys = make_friedman1(n_samples=50, n_features=10, random_state=0)
    sklearn_rfe_selector = sklearn_rfe_selector.fit(Xs, ys)
    sklearn_rfecv_selector = sklearn_rfecv_selector.fit(Xs, ys)

    ## MLJ RFE and RFE with CV
    ## We use the same data and base estimator
    ys = MLJScikitLearnInterface.pyconvert(Vector{Float64}, ys)
    Xs = MLJScikitLearnInterface.pyconvert(Matrix{Float64}, Xs)
    Xs = MLJBase.table(Xs)
    SVR = SVMRegressor
    MLJBase.reports_feature_importances(::SVR) = true
    function MLJBase.feature_importances(::SVR, fitresult, report)
        coef = MLJScikitLearnInterface.pyconvert(Array, fitresult[1].coef_);
        imp = [Pair(Symbol("x$i"), coef[i]) for i in eachindex(coef)]
        return imp
    end
    svm = SVR(kernel = "linear")
    rfe = RecursiveFeatureElimination(model = svm, n_features=5)
    mach = machine(rfe, Xs, ys)
    fit!(mach)
    rfecv = RecursiveFeatureElimination(model = svm)
    tuning_rfe_model  = TunedModel(
        model = rfecv,
        measure = rms,
        tuning = Grid(rng=rng),
        resampling = StratifiedCV(nfolds = 5),
        range = range(rfecv, :n_features, values = 1:10)                                                                                                           
    )
    self_tuning_rfe_mach = machine(tuning_rfe_model, Xs, ys)
    fit!(self_tuning_rfe_mach)

    ## Compare results
    @test report(mach).ranking == MLJScikitLearnInterface.pyconvert(
        Vector{Float64}, sklearn_rfe_selector.ranking_
    )
    @test(
        report(
            self_tuning_rfe_mach
        ).best_report.ranking == MLJScikitLearnInterface.pyconvert(
            Vector{Float64}, sklearn_rfecv_selector.ranking_
        )
    )
end

@testset "serialization for atomic models with non-persistent fitresults" begin
    # https://github.com/alan-turing-institute/MLJ.jl/issues/1099
    X, y = (;x1=rand(8), x2 = rand(8)), categorical(collect("OXXXXOOX"), ordered=true)
    rfe_model = RecursiveFeatureElimination(
        DTM.EphemeralClassifier()
    ) # i.e rfe based on a deterministic_classifier
    mach = MLJBase.machine(rfe_model, X, y)
    MLJBase.fit!(mach, verbosity=0)
    yhat = MLJBase.predict(mach, MLJBase.selectrows(X, 1:2))
    io = IOBuffer()
    MLJBase.save(io, mach)
    seekstart(io)
    mach2 = MLJBase.machine(io)
    close(io)
    @test MLJBase.predict(mach2, (; x1=rand(2), x2 = rand(2))) ==  yhat
end