relu(x) = max.(zero(eltype(x)), x)
sigmoid(x) = one(eltype(x)) ./ (one(eltype(x)) .+ exp.(-x))
identity_fn(x) = x
