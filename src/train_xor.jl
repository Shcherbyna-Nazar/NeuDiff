using DelimitedFiles
include("MyAD.jl")
using .MyAD

# Activation
relu(x) = max.(0, x)

# Define Chain type
struct Chain
    layers::Vector{Any}
end

function Chain(args...)
    return Chain(collect(args))
end

function (chain::Chain)(x)
    for layer in chain.layers
        x = layer(x)
    end
    return x
end

# Load XOR data
X = readdlm("xor_inputs.csv", ',', Float64)
Y = readdlm("xor_targets.csv", ',', Float64)

# Convert inputs/outputs to Variables
function to_variable(x::Array{Float64})
    Variable(x, zeros(size(x)))
end

# Loss: Mean Squared Error
function mse(y_pred::Array{Float64}, y_true::Array{Float64})
    return sum((y_pred .- y_true).^2) / length(y_true)
end

# Derivative of MSE: dL/dy_pred
function mse_grad(y_pred::Array{Float64}, y_true::Array{Float64})
    return 2 .* (y_pred .- y_true) ./ length(y_true)
end

# Model: 2-4-1
model = Chain(
    Dense(2, 4, relu),
    Dense(4, 1, identity)
)

# Forward pass wrapper
function predict(model::Chain, x)
    return model(x)
end

# Parameters
function parameters(model::Chain)
    ps = GraphNode[]
    for layer in model.layers
        push!(ps, layer.W, layer.b)
    end
    return ps
end

# Optimizer: SGD
function update!(params, η)
    for p in params
        p.output .-= η .* p.gradient
    end
end

# Training loop
epochs = 1000
η = 0.1

for epoch in 1:epochs
    x = to_variable(X)
    y = to_variable(Y)

    out = predict(model, x)
    graph = topological_sort(out)

    forward!(graph)

    loss = mse(out.output, y.output)
    out.gradient = mse_grad(out.output, y.output)
    backward!(graph)

    update!(parameters(model), η)

    if epoch % 100 == 0
        println("Epoch $epoch, Loss = $loss")
    end
end

# Final prediction
x = to_variable(X)
y = to_variable(Y)
out = predict(model, x)
graph = topological_sort(out)
forward!(graph)

println("\nFinal predictions:")
println(out.output)
println("Targets:")
println(y.output)
