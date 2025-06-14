push!(LOAD_PATH, "../../src")
using NeuDiff
using .NeuDiff.MyAD, .NeuDiff.MyNN
using BenchmarkTools
using Flux

println("=== Benchmark: MaxPool1D (MyNN vs Flux) ===")

L, C, B = 100, 8, 32
x = randn(Float32, L, C, B)
kernel, stride = 2, 2

mynn_pool = MyNN.MaxPool1D(kernel, stride)
flux_pool = x -> maxpool(x, (kernel,), stride=(stride,))

println("Forward MyNN MaxPool1D:")
@btime begin
    x_var = MyAD.Variable($x, zeros(Float32, size($x)))
    out = mynn_pool(x_var)
    g = MyAD.topological_sort(out)
    MyAD.forward!(g)
end

println("Forward Flux MaxPool1D:")
@btime begin
    $flux_pool($x)
end
