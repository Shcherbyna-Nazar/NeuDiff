using DelimitedFiles
include("MyAD.jl")
using .MyAD

# === Chain structure ===
struct Chain
    layers::Vector{Any}
end

Chain(args...) = Chain(collect(args))

function (chain::Chain)(x)
    for layer in chain.layers
        x = layer(x)
    end
    return x
end

# === Hardcoded XOR data ===
X = [0.0 0.0 1.0 1.0;
     0.0 1.0 0.0 1.0]  # Shape: 2×4

Y = [0.0 1.0 1.0 0.0]  # Shape: 1×4

# === Convert to Variables ===
to_variable(x::Array{Float64}) = Variable(x, zeros(size(x)))

# === Loss: MSE ===
mse(y_pred, y_true) = sum((y_pred .- y_true).^2) / length(y_true)
mse_grad(y_pred, y_true) = 2 .* (y_pred .- y_true) ./ length(y_true)

# === Model ===
model = Chain(
    Dense(2, 8, tanh),
    Dense(8, 1, sigmoid)
)

# === Utility ===
predict(model::Chain, x) = model(x)

function parameters(model::Chain)
    ps = MyAD.GraphNode[]
    for layer in model.layers
        push!(ps, layer.W, layer.b)
    end
    return ps
end

function update!(params, η)
    for p in params
        p.output .-= η .* p.gradient
    end
end

# === Training ===
epochs = 2000
η = 0.1
losses = Float64[]

for epoch in 1:epochs
    x = to_variable(X)
    y = to_variable(Y)

    out = predict(model, x)
    graph = topological_sort(out)

    forward!(graph)
    loss = mse(out.output, y.output)
    push!(losses, loss)

    out.gradient = mse_grad(out.output, y.output)
    backward!(graph, out.gradient)

    update!(parameters(model), η)

    if epoch % 100 == 0
        max_grad = maximum([maximum(abs.(p.gradient)) for p in parameters(model)])
        println("Epoch $epoch, Loss = $(round(loss, digits=5)), MaxGrad = $(round(max_grad, digits=5))")
    end
end

# === Final prediction ===
x = to_variable(X)
out = predict(model, x)
graph = topological_sort(out)
forward!(graph)

println("\nFinal predictions:")
println(round.(out.output, digits=4))
println("Targets:")
println(Y)

# === Save loss to CSV ===
writedlm("xor_training_loss.csv", losses, ',')
