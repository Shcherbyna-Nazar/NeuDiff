include("../../src/MyAD.jl")
include("../../src/MyNN.jl")
using .MyAD, .MyNN
using BenchmarkTools
using Flux

println("=== Benchmark: Embedding + Flatten + Dense (MyNN vs Flux) ===")

vocab_size, emb_dim, seq_len, batch_size = 10000, 64, 20, 32
indices = rand(1:vocab_size, seq_len, batch_size)
y_true = randn(Float32, 10, batch_size)

mynn_emb = MyNN.Embedding(vocab_size, emb_dim)
mynn_dense = MyNN.Dense(seq_len * emb_dim, 10)
mynn_chain = MyNN.Chain(x -> mynn_emb(x), flatten_last_two_dims, mynn_dense)

flux_emb = Flux.Embedding(vocab_size, emb_dim)
flux_dense = Flux.Dense(seq_len * emb_dim, 10)
flux_chain = Flux.Chain(flux_emb, x -> reshape(x, :, size(x, 3)), flux_dense)

println("Forward MyNN Embedding+Flatten+Dense:")
@btime begin
    y_pred = mynn_chain($indices)
    g = MyAD.topological_sort(y_pred)
    MyAD.forward!(g)
end

println("Forward Flux Embedding+Flatten+Dense:")
@btime begin
    $flux_chain($indices)
end
