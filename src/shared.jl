## Pkg Traits

for M in [
    :FeatureSelector,
    :DeterministicRecursiveFeatureElimination,
    :ProbabilisticRecursiveFeatureElimination,
    ]
    quote
        MMI.package_name(::Type{<:$M})  = "FeatureSelection"
        MMI.package_uuid(::Type{<:$M})  = "33837fe5-dbff-4c9e-8c2f-c5612fe2b8b6"
        MMI.package_url(::Type{<:$M})   = "https://github.com/JuliaAI/FeatureSelection.jl"
        MMI.is_pure_julia(::Type{<:$M}) = true
        MMI.package_license(::Type{<:$M})   = "MIT"
    end |> eval
end

