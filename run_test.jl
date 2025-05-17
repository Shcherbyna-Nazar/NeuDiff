include("src/MyAD.jl")
using .MyAD

x = Variable(5.0, 0.0)
y = sin(x * x)
graph = topological_sort(y)
forward!(graph)
backward!(graph)

println("f(x) = ", y.output)
println("df/dx = ", x.gradient)
