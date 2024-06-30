var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"CurrentModule = FeatureSelection","category":"page"},{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/#Models","page":"API","title":"Models","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"FeatureSelector\nRecursiveFeatureElimination","category":"page"},{"location":"api/#FeatureSelection.FeatureSelector","page":"API","title":"FeatureSelection.FeatureSelector","text":"FeatureSelector\n\nA model type for constructing a feature selector, based on FeatureSelection.jl, and implementing the MLJ model interface.\n\nFrom MLJ, the type can be imported using\n\nFeatureSelector = @load FeatureSelector pkg=FeatureSelection\n\nDo model = FeatureSelector() to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in FeatureSelector(features=...).\n\nUse this model to select features (columns) of a table, usually as part of a model Pipeline.\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with\n\nmach = machine(model, X)\n\nwhere\n\nX: any table of input features, where \"table\" is in the sense of Tables.jl\n\nTrain the machine using fit!(mach, rows=...).\n\nHyper-parameters\n\nfeatures: one of the following, with the behavior indicated:\n[] (empty, the default): filter out all features (columns) which were not encountered in training\nnon-empty vector of feature names (symbols): keep only the specified features (ignore=false) or keep only unspecified features (ignore=true)\nfunction or other callable: keep a feature if the callable returns true on its name. For example, specifying FeatureSelector(features = name -> name in [:x1, :x3], ignore = true) has the same effect as FeatureSelector(features = [:x1, :x3], ignore = true), namely to select all features, with the exception of :x1 and :x3.\nignore: whether to ignore or keep specified features, as explained above\n\nOperations\n\ntransform(mach, Xnew): select features from the table Xnew as specified by the model, taking features seen during training into account, if relevant\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\nfeatures_to_keep: the features that will be selected\n\nExample\n\nusing MLJ\n\nX = (ordinal1 = [1, 2, 3],\n     ordinal2 = coerce([\"x\", \"y\", \"x\"], OrderedFactor),\n     ordinal3 = [10.0, 20.0, 30.0],\n     ordinal4 = [-20.0, -30.0, -40.0],\n     nominal = coerce([\"Your father\", \"he\", \"is\"], Multiclass));\n\nselector = FeatureSelector(features=[:ordinal3, ], ignore=true);\n\njulia> transform(fit!(machine(selector, X)), X)\n(ordinal1 = [1, 2, 3],\n ordinal2 = CategoricalValue{Symbol,UInt32}[\"x\", \"y\", \"x\"],\n ordinal4 = [-20.0, -30.0, -40.0],\n nominal = CategoricalValue{String,UInt32}[\"Your father\", \"he\", \"is\"],)\n\n\n\n\n\n\n","category":"type"},{"location":"api/#FeatureSelection.RecursiveFeatureElimination","page":"API","title":"FeatureSelection.RecursiveFeatureElimination","text":"RecursiveFeatureElimination(model, n_features, step)\n\nThis model implements a recursive feature elimination algorithm for feature selection. It recursively removes features, training a base model on the remaining features and evaluating their importance until the desired number of features is selected.\n\nConstruct an instance with default hyper-parameters using the syntax rfe_model = RecursiveFeatureElimination(model=...). Provide keyword arguments to override hyper-parameter defaults.\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance rfe_model to data with\n\nmach = machine(rfe_model, X, y)\n\nOR, if the base model supports weights, as\n\nmach = machine(rfe_model, X, y, w)\n\nHere:\n\nX is any table of input features (eg, a DataFrame) whose columns are of the scitype as that required by the base model; check column scitypes with schema(X) and column scitypes required by base model with input_scitype(basemodel).\ny is the target, which can be any table of responses whose element scitype is   Continuous or Finite depending on the target_scitype required by the base model;   check the scitype with scitype(y).\nw is the observation weights which can either be nothing(default) or an AbstractVector whoose element scitype is Count or Continuous. This is different from weights kernel which is an hyperparameter to the model, see below.\n\nTrain the machine using fit!(mach, rows=...).\n\nHyper-parameters\n\nmodel: A base model with a fit method that provides information on feature feature importance (i.e reports_feature_importances(model) == true)\nn_features::Real = 0: The number of features to select. If 0, half of the features are selected. If a positive integer, the parameter is the absolute number of features to select. If a real number between 0 and 1, it is the fraction of features to select.\nstep::Real=1: If the value of step is at least 1, it signifies the quantity of features to eliminate in each iteration. Conversely, if step falls strictly within the range of 0.0 to 1.0, it denotes the proportion (rounded down) of features to remove during each iteration.\n\nOperations\n\ntransform(mach, X): transform the input table X into a new table containing only\n\ncolumns corresponding to features gotten from the RFE algorithm.\n\npredict(mach, X): transform the input table X into a new table same as in\ntransform(mach, X) above and predict using the fitted base model on the transformed table.\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\nfeatures_left: names of features remaining after recursive feature elimination.\nmodel_fitresult: fitted parameters of the base model.\n\nReport\n\nThe fields of report(mach) are:\n\nranking: The feature ranking of each features in the training dataset.\nmodel_report: report for the fitted base model.\nfeatures: names of features seen during the training process.\n\nExamples\n\nusing FeatureSelection, MLJ, StableRNGs\n\nRandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree\n\n# Creates a dataset where the target only depends on the first 5 columns of the input table.\nA = rand(rng, 50, 10);\ny = 10 .* sin.(\n        pi .* A[:, 1] .* A[:, 2]\n    ) + 20 .* (A[:, 3] .- 0.5).^ 2 .+ 10 .* A[:, 4] .+ 5 * A[:, 5]);\nX = MLJ.table(A);\n\n# fit a rfe model\nrf = RandomForestRegressor()\nselector = RecursiveFeatureElimination(model = rf)\nmach = machine(selector, X, y)\nfit!(mach)\n\n# view the feature importances\nfeature_importances(mach)\n\n# predict using the base model\nXnew = MLJ.table(rand(rng, 50, 10));\npredict(mach, Xnew)\n\n\n\n\n\n\n","category":"function"},{"location":"#FeatureSelection","page":"Home","title":"FeatureSelection","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"FeatureSelction is a julia package containing implementations of feature selection algorithms for use with the machine learning toolbox MLJ.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"On a running instance of Julia with at least version 1.6 run","category":"page"},{"location":"","page":"Home","title":"Home","text":"import Pkg;\nPkg.add(\"FeatureSelection\")","category":"page"},{"location":"#Example-Usage","page":"Home","title":"Example Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Lets build a supervised recursive feature eliminator with RandomForestRegressor  from DecisionTree.jl as our base model. But first we need a dataset to train on. We shall create a synthetic dataset popularly  known in the R community as the friedman dataset#1. Notice how the target vector for this  dataset depends on only the first five columns of feature table. So we expect that our  recursive feature elimination should return the first columns as important features.","category":"page"},{"location":"","page":"Home","title":"Home","text":"DocTestSetup = quote\n  using MLJ, FeatureSelection, StableRNGs\n  rng = StableRNG(10)\n  A = rand(rng, 50, 10)\n  X = MLJ.table(A) # features\n  y = @views(\n      10 .* sin.(\n          pi .* A[:, 1] .* A[:, 2]\n      ) .+ 20 .* (A[:, 3] .- 0.5).^ 2 .+ 10 .* A[:, 4] .+ 5 * A[:, 5]\n  ) # target\n  RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree\n  forest = RandomForestRegressor(rng=rng)\n  rfe = RecursiveFeatureElimination(\n    model = forest, n_features=5, step=1\n  ) # see doctring for description of defaults  \n  mach = machine(rfe, X, y)\n  fit!(mach)\n\n  rfe = RecursiveFeatureElimination(model = forest)\n  tuning_rfe_model  = TunedModel(\n      model = rfe,\n      measure = rms,\n      tuning = Grid(rng=rng),\n      resampling = StratifiedCV(nfolds = 5),\n      range = range(\n          rfe, :n_features, values = 1:10\n      )\n  )\n  self_tuning_rfe_mach = machine(tuning_rfe_model, X, y)\n  fit!(self_tuning_rfe_mach)\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"using MLJ, FeatureSelection, StableRNGs\nrng = StableRNG(10)\nA = rand(rng, 50, 10)\nX = MLJ.table(A) # features\ny = @views(\n    10 .* sin.(\n        pi .* A[:, 1] .* A[:, 2]\n    ) .+ 20 .* (A[:, 3] .- 0.5).^ 2 .+ 10 .* A[:, 4] .+ 5 * A[:, 5]\n) # target","category":"page"},{"location":"","page":"Home","title":"Home","text":"Now we that we have our data, we can create our recursive feature elimination model and  train it on our dataset","category":"page"},{"location":"","page":"Home","title":"Home","text":"RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree\nforest = RandomForestRegressor(rng=rng)\nrfe = RecursiveFeatureElimination(\n    model = forest, n_features=5, step=1\n) # see doctring for description of defaults\nmach = machine(rfe, X, y)\nfit!(mach)","category":"page"},{"location":"","page":"Home","title":"Home","text":"We can inspect the feature importances in two ways:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> report(mach).ranking\n10-element Vector{Int64}:\n 1\n 1\n 1\n 1\n 1\n 2\n 3\n 4\n 5\n 6\n\njulia> feature_importances(mach)\n10-element Vector{Pair{Symbol, Int64}}:\n  :x1 => 6\n  :x2 => 5\n  :x3 => 4\n  :x4 => 3\n  :x5 => 2\n  :x6 => 1\n  :x7 => 1\n  :x8 => 1\n  :x9 => 1\n :x10 => 1","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that a variable with lower rank has more significance than a variable with higher rank; while a variable with higher feature importance is better than a variable with lower feature importance.","category":"page"},{"location":"","page":"Home","title":"Home","text":"We can view the important features used by our model by inspecting the fitted_params  object.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> p = fitted_params(mach)\n(features_left = [:x1, :x2, :x3, :x4, :x5],\n model_fitresult = (forest = Ensemble of Decision Trees\nTrees:      100\nAvg Leaves: 25.26\nAvg Depth:  8.36,),)\n\njulia> p.features_left\n5-element Vector{Symbol}:\n :x1\n :x2\n :x3\n :x4\n :x5","category":"page"},{"location":"","page":"Home","title":"Home","text":"We can also call the predict method on the fitted machine, to predict using a  random forest regressor trained using only the important features, or call the transform  method, to select just those features from some new table including all the original  features. For more info, type ?RecursiveFeatureElimination on a Julia REPL.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Okay, let's say that we didn't know that our synthetic dataset depends on only five  columns from our feature table. We could apply cross fold validation  StratifiedCV(nfolds=5) with our recursive feature elimination model to select the  optimal value of n_features for our model. In this case we will use a simple Grid  search with root mean square as the measure. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"rfe = RecursiveFeatureElimination(model = forest)\ntuning_rfe_model  = TunedModel(\n    model = rfe,\n    measure = rms,\n    tuning = Grid(rng=rng),\n    resampling = StratifiedCV(nfolds = 5),\n    range = range(\n        rfe, :n_features, values = 1:10\n    )\n)\nself_tuning_rfe_mach = machine(tuning_rfe_model, X, y)\nfit!(self_tuning_rfe_mach)","category":"page"},{"location":"","page":"Home","title":"Home","text":"As before we can inspect the important features by inspecting the object returned by  fitted_params or feature_importances as shown below.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> fitted_params(self_tuning_rfe_mach).best_fitted_params.features_left\n5-element Vector{Symbol}:\n :x1\n :x2\n :x3\n :x4\n :x5\n\njulia> feature_importances(self_tuning_rfe_mach)\n10-element Vector{Pair{Symbol, Int64}}:\n  :x1 => 6\n  :x2 => 5\n  :x3 => 4\n  :x4 => 3\n  :x5 => 2\n  :x6 => 1\n  :x7 => 1\n  :x8 => 1\n  :x9 => 1\n :x10 => 1","category":"page"},{"location":"","page":"Home","title":"Home","text":"and call predict on the tuned model machine as shown below","category":"page"},{"location":"","page":"Home","title":"Home","text":"Xnew = MLJ.table(rand(rng, 50, 10)) # create test data\npredict(self_tuning_rfe_mach, Xnew)","category":"page"},{"location":"","page":"Home","title":"Home","text":"In this case, prediction is done using the best recursive feature elimination model gotten  from the tuning process above.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For resampling methods different from cross-validation, and for other  TunedModel options, such as parallelization, see the   Tuning Models section of the MLJ manual. MLJ Documentation","category":"page"},{"location":"","page":"Home","title":"Home","text":"DocTestSetup = nothing","category":"page"}]
}
