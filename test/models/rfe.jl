using .DummyTestModels
const DTM = DummyTestModels

@testset "RecursiveFeatureElimination" begin
    # Data for use in testset
    X = rand(rng, 50, 10)
    y = @views 10 .* sin.(pi .* X[:, 1] .* X[:, 2]) + 20 .* (X[:, 3] .- 0.5).^2 .+ 10 .* X[:, 4] .+ 5 * X[:, 5]
    yc = categorical(Int.(y .> mean(y)))
    Xt = MLJBase.table(X)
    Xnew = MLJBase.table(rand(rng, 50, 10))
    Xnew2 = MLJBase.table(rand(rng, 50, 10), names = [Symbol("y$i") for i in 1:10])

    # Constructor tests
    @test_throws FeatureSelection.ERR_SPECIFY_MODEL RecursiveFeatureElimination()
    reg = DTM.DeterministicConstantRegressor()
    @test_throws FeatureSelection.ERR_FEATURE_IMPORTANCE_SUPPORT RecursiveFeatureElimination(model=DTM.DeterministicConstantRegressor())

    # RandomForest models with copied random state
    rf = MLJDecisionTreeInterface.RandomForestRegressor(rng=copy(rng))
    rf2 = MLJDecisionTreeInterface.RandomForestClassifier(rng=copy(rng))
    rf3 = MLJDecisionTreeInterface.RandomForestRegressor(rng=copy(rng))

    # Test logging of warnings
    @test_logs(
        (
            :warn, "Using `model=$rf`. Ignoring keyword specification `model=$rf`. "
        ),
        RecursiveFeatureElimination(rf, model = rf)
    )
    @test_logs(
        (
            :warn, "Specified `step` must be greater than zero.\nResetting `step = 1`\n"
        ),
        RecursiveFeatureElimination(model = rf, step=-1)
    )
    @test_logs(
        (
            :warn,
            "Specified `n_features` must be non-negative.\nResetting `n_features = 0`\n"
        ),
        RecursiveFeatureElimination(model = rf, n_features=-5)
    )
    @test_logs(
        (
            :warn, "Specified `step` must be greater than zero.\nResetting `step = 1`\nSpecified `n_features` must be non-negative.\nResetting `n_features = 0`\n",
        ),
        RecursiveFeatureElimination(model = rf, step = 0, n_features=-1)
    )

    # Check instance types
    selector = RecursiveFeatureElimination(model=rf)
    selector2 = RecursiveFeatureElimination(model=rf2)
    @test selector isa FeatureSelection.DeterministicRecursiveFeatureElimination
    @test selector2 isa FeatureSelection.ProbabilisticRecursiveFeatureElimination
    @test MLJBase.constructor(selector) == RecursiveFeatureElimination

    # Fit models
    selector3 = RecursiveFeatureElimination(model=rf3, n_features=0.5, step=1e-6)
    selector4 = RecursiveFeatureElimination(model=rf, n_features=11)
    selector_mach = machine(selector, Xt, y)
    selector_mach2 = machine(selector2, Xt, yc)
    selector_mach3 = machine(selector3, Xt, y)
    selector_mach4 = machine(selector4, Xt, y)

    fit!(selector_mach)
    fit!(selector_mach2)
    fit!(selector_mach3)
    @test_logs(
        (:warn, "n_features > number of features in training data, hence no feature will be eliminated."),
        match_mode=:any,
        fit!(selector_mach4)
    )

    # Check fitted parameters
    selector_fp = fitted_params(selector_mach)
    selector_fp2 = fitted_params(selector_mach2)
    selector_fp3 = fitted_params(selector_mach3)

    @test propertynames(selector_fp) == (:features_left, :model_fitresult)
    @test selector_fp.features_left == [:x4, :x2, :x1, :x5, :x3]
    @test selector_fp2.features_left == [:x4, :x2, :x1, :x3, :x10]
    @test selector_fp3.features_left == selector_fp.features_left
    @test selector_fp.model_fitresult == MLJBase.fitted_params(selector_mach.model.model, selector_mach.fitresult.model_fitresult)

    @test feature_importances(selector_mach) == [:x9 => 1, :x2 => 6, :x5 => 6, :x6 => 2, :x7 => 3, :x3 => 6, :x8 => 4, :x4 => 6, :x10 => 5, :x1 => 6]
    @test feature_importances(selector_mach3) == feature_importances(selector_mach)

    # Check report
    rpt = report(selector_mach)
    @test rpt.scores == Dict([:x9 => 1, :x2 => 6, :x5 => 6, :x6 => 2, :x7 => 3, :x3 => 6, :x8 => 4, :x4 => 6, :x10 => 5, :x1 => 6])
    @test report(selector_mach3).scores == rpt.scores

    # Predict
    yhat = predict(selector_mach, Xnew)
    @test scitype(yhat) === AbstractVector{Continuous}

    # Transform
    trf = transform(selector_mach, Xnew)
    sch = MLJBase.schema(trf)
    @test sch.names === (:x4, :x2, :x1, :x5, :x3)
    @test sch.scitypes === (Continuous, Continuous, Continuous, Continuous, Continuous)
    @test_throws FeatureSelection.ERR_FEATURES_SEEN transform(selector_mach, Xnew2)

    # Traits
    @test MLJBase.package_name(selector) == "FeatureSelection"
    @test MLJBase.load_path(selector) == "FeatureSelection.RecursiveFeatureElimination"
    @test MLJBase.iteration_parameter(selector) == FeatureSelection.prepend(:model, MLJBase.iteration_parameter(selector.model))
    @test MLJBase.training_losses(selector, rpt) == MLJBase.training_losses(selector.model, rpt.model_report)
end

@testset "Compare results for RFE with scikit-learn" begin
    # Import necessary modules from scikit-learn using MLJScikitLearnInterface
    sklearn = MLJScikitLearnInterface.pyimport("sklearn")
    make_friedman1 = MLJScikitLearnInterface.pyimport("sklearn.datasets"=>"make_friedman1")
    RFE_sklearn = MLJScikitLearnInterface.pyimport("sklearn.feature_selection"=>"RFE")
    RFECV_sklearn = MLJScikitLearnInterface.pyimport("sklearn.feature_selection"=>"RFECV")
    SVR_sklearn = MLJScikitLearnInterface.pyimport("sklearn.svm"=>"SVR")

    # Initialize the SVR estimator
    sklearn_svr_estimator = SVR_sklearn(kernel="linear")

    # Initialize RFE and RFECV selectors
    sklearn_rfe_selector = RFE_sklearn(sklearn_svr_estimator, n_features_to_select=5, step=1)
    sklearn_rfecv_selector = RFECV_sklearn(sklearn_svr_estimator, step=1, cv=5)

    # Generate sample data
    Xs, ys = make_friedman1(n_samples=50, n_features=10, random_state=0)

    # Fit RFE and RFECV selectors
    sklearn_rfe_selector.fit(Xs, ys)
    sklearn_rfecv_selector.fit(Xs, ys)

    # Convert data to Julia types and create an MLJ table
    ys = MLJScikitLearnInterface.pyconvert(Vector{Float64}, ys)
    Xs = MLJScikitLearnInterface.pyconvert(Matrix{Float64}, Xs)
    Xs = MLJBase.table(Xs)
    feature_names = schema(Xs).names

    # Extend SVR regressor from MLJScikitLearnInterface to support `feature_importances`
    SVR = SVMRegressor
    MLJBase.reports_feature_importances(::SVR) = true
    function MLJBase.feature_importances(::SVR, fitresult, report)
        coef = MLJScikitLearnInterface.pyconvert(Array, fitresult[1].coef_)
        imp = [Pair(report.names[i], coef[i]) for i in eachindex(coef)]
        return imp
    end

    # Train MLJ RFE and RFECV models
    svm = SVR(kernel="linear")
    rfe = RecursiveFeatureElimination(model=svm, n_features=5)
    mach = machine(rfe, Xs, ys)
    fit!(mach)

    rfecv = RecursiveFeatureElimination(model=svm)
    tuning_rfe_model = TunedModel(
        model=rfecv,
        measure=rms,
        tuning=Grid(rng=rng),
        resampling=StratifiedCV(nfolds=5),
        range=range(rfecv, :n_features, values=1:10)
    )
    self_tuning_rfe_mach = machine(tuning_rfe_model, Xs, ys)
    fit!(self_tuning_rfe_mach)

    # Compare results
    # Convert MLJ RFE scores to rankings
    mlj_rfe_scores = report(mach).scores
    m = maximum(values(mlj_rfe_scores))
    mlj_rfe_ranking = [m - mlj_rfe_scores[ftr] + 1 for ftr in feature_names]
    @test mlj_rfe_ranking == MLJScikitLearnInterface.pyconvert(Vector{Float64}, sklearn_rfe_selector.ranking_)

    mlj_tuned_rfe_scores = report(self_tuning_rfe_mach).best_report.scores
    m = maximum(values(mlj_tuned_rfe_scores))
    mlj_tuned_rfe_ranking = [m - mlj_tuned_rfe_scores[ftr] + 1 for ftr in feature_names]
    @test mlj_tuned_rfe_ranking == MLJScikitLearnInterface.pyconvert(Vector{Float64}, sklearn_rfecv_selector.ranking_)
end

@testset "Serialization for atomic models with non-persistent fitresults" begin
    # https://github.com/alan-turing-institute/MLJ.jl/issues/1099
    X, y = (; x1=rand(8), x2=rand(8)), categorical(collect("OXXXXOOX"), ordered=true)
    rfe_model = RecursiveFeatureElimination(DTM.EphemeralClassifier()) # RFE based on a deterministic classifier
    mach = machine(rfe_model, X, y)
    fit!(mach, verbosity=0)

    yhat = predict(mach, selectrows(X, 1:2))
    io = IOBuffer()
    MLJBase.save(io, mach)
    seekstart(io)
    mach2 = machine(io)
    close(io)

    @test predict(mach2, (; x1=rand(2), x2=rand(2))) == yhat
end
