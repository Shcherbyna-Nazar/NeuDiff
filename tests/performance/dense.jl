push!(LOAD_PATH, "../../src")
using NeuDiff
using .NeuDiff.MyAD, .NeuDiff.MyNN
using BenchmarkTools
using Flux

println("=== Benchmark: Dense Layer (MyNN vs Flux) ===")

in_dim, hid_dim, out_dim, batch_size = 128, 64, 10, 32
x = randn(Float32, in_dim, batch_size)
y_true = randn(Float32, out_dim, batch_size)

# MyNN model
mynn_model = MyNN.Chain(
    MyNN.Dense(in_dim, hid_dim, relu),
    MyNN.Dense(hid_dim, out_dim, identity_fn)
)

# Flux model
flux_model = Flux.Chain(
    Flux.Dense(in_dim, hid_dim, relu),
    Flux.Dense(hid_dim, out_dim)
)

function mynn_loss(x, y)
    x_var = MyAD.Variable(x, zeros(Float32, size(x)))
    y_pred = mynn_model(x_var)
    graph = MyAD.topological_sort(y_pred)
    MyAD.forward!(graph)
    loss = sum((y_pred.output .- y).^2) / length(y)
    return loss, y_pred, graph
end

function mynn_backward!(y_pred, y_true, graph)
    grad_loss = 2 * (y_pred.output .- y_true) / length(y_true)
    MyAD.backward!(graph, grad_loss)
end

function flux_loss(x, y)
    y_pred = flux_model(x)
    sum((y_pred .- y).^2) / length(y)
end

println("Forward + Backward MyNN:")
@btime begin
    MyNN.zero_gradients!(mynn_model)
    loss, y_pred, graph = mynn_loss($x, $y_true)
    mynn_backward!(y_pred, $y_true, graph)
end

println("Forward + Backward Flux:")
@btime begin
    gs = Flux.gradient(() -> flux_loss($x, $y_true), Flux.params(flux_model))
end
