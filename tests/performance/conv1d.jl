include("../../src/MyAD.jl")
include("../../src/MyNN.jl")
using .MyAD, .MyNN
using BenchmarkTools
using Flux

println("=== Benchmark: Conv1D Layer (MyNN vs Flux) ===")

L, C, B, K, O = 128, 16, 32, 3, 32
x = randn(Float32, L, C, B)
W = randn(Float32, K, C, O)
b = randn(Float32, O, 1)

# MyNN Conv1D model
mynn_conv = MyNN.Conv1D(C, O, K)
mynn_conv.W.output .= W
mynn_conv.b.output .= b

# Flux Conv1D model
flux_conv = Flux.Conv((K,), C => O, identity)
flux_conv.weight .= W
flux_conv.bias .= vec(b)

println("Forward MyNN Conv1D:")
@btime begin
    x_var = MyAD.Variable($x, zeros(Float32, size($x)))
    out = mynn_conv(x_var)
    g = MyAD.topological_sort(out)
    MyAD.forward!(g)
end

println("Forward Flux Conv1D:")
@btime begin
    $flux_conv($x)
end

println("Backward MyNN Conv1D:")
@btime begin
    x_var = MyAD.Variable($x, zeros(Float32, size($x)))
    out = mynn_conv(x_var)
    g = MyAD.topological_sort(out)
    MyAD.forward!(g)
    MyAD.backward!(g, ones(Float32, size(out.output)))
end

println("Backward Flux Conv1D:")
@btime begin
    gs = Flux.gradient(() -> sum($flux_conv($x)), Flux.params(flux_conv))
end
