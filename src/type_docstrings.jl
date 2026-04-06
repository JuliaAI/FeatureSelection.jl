# This file cannot be include before types and all metadata is defined

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
