include("../../src/MyAD.jl")
include("../../src/MyNN.jl")
using .MyAD, .MyNN
using Flux
using BenchmarkTools
using .MyNN: relu, identity_fn, flatten_last_two_dims

println("=== Benchmark: Złożony model (Embedding -> Conv1D -> MaxPool1D -> Flatten -> Dense -> Dense) ===")

# Parametry modelu
vocab_size, emb_dim, seq_len, batch_size = 3000, 32, 50, 16
in_channels = emb_dim
conv_out_channels = 64
kernel_size = 3
pool_kernel, pool_stride = 2, 2
flattened_dim = ((seq_len - kernel_size + 1) ÷ pool_stride) * conv_out_channels
hidden_dim = 64
output_dim = 8

# Dane wejściowe
indices = rand(1:vocab_size, seq_len, batch_size)
y_true = randn(Float32, output_dim, batch_size)

# --- Inicjalizacja warstw MyNN ---
embed_layer = MyNN.Embedding(vocab_size, emb_dim)
conv_layer = MyNN.Conv1D(in_channels, conv_out_channels, kernel_size, relu)
dense1 = MyNN.Dense(flattened_dim, hidden_dim, relu)
dense2 = MyNN.Dense(hidden_dim, output_dim, identity_fn)

mynn_model = MyNN.Chain(
    embed_layer,
    x -> PermuteDimsOp(x, (2, 1, 3)),  # (L, C, B) -> (C, L, B)
    conv_layer,
    MyNN.MaxPool1D(pool_kernel, pool_stride),
    flatten_last_two_dims,
    dense1,
    dense2
)

function mynn_fullpass()
    x_var = indices
    y_pred = mynn_model(x_var)
    graph = MyAD.topological_sort(y_pred)
    MyAD.forward!(graph)
    loss = sum((y_pred.output .- y_true).^2) / length(y_true)
    grad_loss = 2 * (y_pred.output .- y_true) / length(y_true)
    MyAD.backward!(graph, grad_loss)
end

# --- Inicjalizacja warstw Flux ---
flux_embed = Flux.Embedding(vocab_size, emb_dim)
flux_conv = Flux.Conv((kernel_size,), emb_dim => conv_out_channels, relu; stride=1, pad=0)
flux_dense1 = Flux.Dense(flattened_dim, hidden_dim, relu)
flux_dense2 = Flux.Dense(hidden_dim, output_dim)

flux_model = Flux.Chain(
    flux_embed,
    x -> permutedims(x, (2, 1, 3)),  # (L, C, B) -> (C, L, B)
    flux_conv,
    x -> maxpool(x, (pool_kernel,), stride=(pool_stride,)),
    x -> reshape(x, :, size(x, 3)),
    flux_dense1,
    flux_dense2
)

function flux_fullpass()
    y_pred = flux_model(indices)
    loss = sum((y_pred .- y_true).^2) / length(y_true)
    gs = Flux.gradient(() -> sum((flux_model(indices) .- y_true).^2) / length(y_true), Flux.params(flux_model))
end

# --- Benchmark MyNN ---
println("MyNN: Forward + Backward + Grad")
@btime mynn_fullpass()

# --- Benchmark Flux ---
println("Flux: Forward + Backward + Grad")
@btime flux_fullpass()
