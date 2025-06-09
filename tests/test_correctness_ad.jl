using Test
include("../src/MyAD.jl")

using .MyAD
using Zygote
using Flux
using .MyAD:relu, sigmoid, identity_fn, tanh, broadcast_add

@testset "Scalar addition (a + b)" begin
    a_val, b_val = 3.0f0, 2.0f0

    # Zygote ground truth
    f_zyg(a, b) = a + b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

    # MyAD
    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    out = ScalarOperator(+, a, b)
    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output[1], y_zyg; atol=1e-6)

    backward!(graph, [1.0f0])
    @test isapprox(a.gradient[1], grad_zyg[1]; atol=1e-6)
    @test isapprox(b.gradient[1], grad_zyg[2]; atol=1e-6)
end

@testset "Scalar multiplication (a * b)" begin
    a_val, b_val = 3.0f0, 2.0f0

    # Zygote ground truth
    f_zyg(a, b) = a * b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

    # MyAD
    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    out = ScalarOperator(*, a, b)
    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output[1], y_zyg; atol=1e-6)

    backward!(graph, [1.0f0])
    @test isapprox(a.gradient[1], grad_zyg[1]; atol=1e-6)
    @test isapprox(b.gradient[1], grad_zyg[2]; atol=1e-6)
end

@testset "Scalar division (a / b)" begin
    a_val, b_val = 6.0f0, 2.0f0

    # Zygote ground truth
    f_zyg(a, b) = a / b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

    # MyAD
    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    out = ScalarOperator(/, a, b)
    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output[1], y_zyg; atol=1e-6)

    backward!(graph, [1.0f0])
    @test isapprox(a.gradient[1], grad_zyg[1]; atol=1e-6)
    @test isapprox(b.gradient[1], grad_zyg[2]; atol=1e-6)
end

@testset "Scalar subtraction (a - b)" begin
    a_val, b_val = 5.0f0, 2.0f0

    f_zyg(a, b) = a - b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    out = ScalarOperator(-, a, b)
    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output[1], y_zyg; atol=1e-6)

    backward!(graph, [1.0f0])
    @test isapprox(a.gradient[1], grad_zyg[1]; atol=1e-6)
    @test isapprox(b.gradient[1], grad_zyg[2]; atol=1e-6)
end

@testset "Matrix multiplication (A * B)" begin
    A_val = randn(Float32, 2, 3)
    B_val = randn(Float32, 3, 1)

    # Zygote ground truth
    f_zyg(A, B) = A * B
    y_zyg, back_zyg = Zygote.pullback(f_zyg, A_val, B_val)
    grad_zyg = back_zyg(ones(Float32, 2, 1))

    # MyAD
    A = Variable(A_val, zeros(Float32, size(A_val)))
    B = Variable(B_val, zeros(Float32, size(B_val)))
    out = MatMulOperator(A, B)
    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output, y_zyg; atol=1e-5)

    backward!(graph, ones(Float32, 2, 1))
    @test isapprox(A.gradient, grad_zyg[1]; atol=1e-5)
    @test isapprox(B.gradient, grad_zyg[2]; atol=1e-5)
end


@testset "ReLU activation" begin
    x_val = Float32.([-1.0, 0.0, 1.0, 2.0])
    
    f_zyg(x) = relu.(x)
    y_zyg, back_zyg = Zygote.pullback(f_zyg, x_val)
    grad_zyg = back_zyg(ones(Float32, 4))

    x = Variable(x_val, zeros(Float32, 4))
    z = BroadcastedOperator(relu, x)
    graph = topological_sort(z)

    forward!(graph)
    @test isapprox(z.output, y_zyg; atol=1e-6)

    backward!(graph, ones(Float32, 4))
    @test isapprox(x.gradient, grad_zyg[1]; atol=1e-6)
end

@testset "Sigmoid activation" begin
    x_val = Float32.([-1.0, 0.0, 1.0, 2.0])
    
    f_zyg(x) = sigmoid.(x)
    y_zyg, back_zyg = Zygote.pullback(f_zyg, x_val)
    grad_zyg = back_zyg(ones(Float32, 4))

    x = Variable(x_val, zeros(Float32, 4))
    z = BroadcastedOperator(sigmoid, x)
    graph = topological_sort(z)

    forward!(graph)
    @test isapprox(z.output, y_zyg; atol=1e-6)

    backward!(graph, ones(Float32, 4))
    @test isapprox(x.gradient, grad_zyg[1]; atol=1e-6)
end

@testset "Identity activation" begin
    x_val = Float32.([-1.0, 0.0, 1.0, 2.0])
    
    f_zyg(x) = identity.(x)
    y_zyg, back_zyg = Zygote.pullback(f_zyg, x_val)
    grad_zyg = back_zyg(ones(Float32, 4))

    x = Variable(x_val, zeros(Float32, 4))
    z = BroadcastedOperator(identity_fn, x)
    graph = topological_sort(z)

    forward!(graph)
    @test isapprox(z.output, y_zyg; atol=1e-6)

    backward!(graph, ones(Float32, 4))
    @test isapprox(x.gradient, grad_zyg[1]; atol=1e-6)
end

@testset "Tanh activation" begin
    x_val = Float32.([-1.0, 0.0, 1.0, 2.0])
    
    f_zyg(x) = tanh.(x)
    y_zyg, back_zyg = Zygote.pullback(f_zyg, x_val)
    grad_zyg = back_zyg(ones(Float32, 4))

    x = Variable(x_val, zeros(Float32, 4))
    z = BroadcastedOperator(tanh, x)
    graph = topological_sort(z)

    forward!(graph)
    @test isapprox(z.output, y_zyg; atol=1e-6)

    backward!(graph, ones(Float32, 4))
    @test isapprox(x.gradient, grad_zyg[1]; atol=1e-6)
end

@testset "Dense layer-like composition (sigmoid(A*x + b))" begin
    A_val = randn(Float32, 4, 3)
    x_val = randn(Float32, 3, 1)
    b_val = randn(Float32, 4, 1)

    f_zyg(A, x, b) = sigmoid.(A * x .+ b)
    y_zyg, back_zyg = Zygote.pullback(f_zyg, A_val, x_val, b_val)
    grad_zyg = back_zyg(ones(Float32, 4, 1))

    A = Variable(A_val, zeros(Float32, size(A_val)))
    x = Variable(x_val, zeros(Float32, size(x_val)))
    b = Variable(b_val, zeros(Float32, size(b_val)))

    z1 = MatMulOperator(A, x)
    z2 = ScalarOperator(broadcast_add, z1, b)
    out = BroadcastedOperator(sigmoid, z2)

    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output, y_zyg; atol=1e-5)

    backward!(graph, ones(Float32, 4, 1))
    @test isapprox(A.gradient, grad_zyg[1]; atol=1e-5)
    @test isapprox(x.gradient, grad_zyg[2]; atol=1e-5)
    @test isapprox(b.gradient, grad_zyg[3]; atol=1e-5)
end

@testset "Flatten last two dims operation with batch" begin
    x_val = reshape(Float32.(1:24), (2, 3, 4))  # shape: (L=2, C=3, B=4)

    f_zyg(x) = reshape(x, :, size(x, ndims(x)))  # shape: (6, 4)
    y_zyg, back_zyg = Zygote.pullback(f_zyg, x_val)
    grad_zyg = back_zyg(ones(Float32, 6, 4))

    # MyAD
    x = Variable(x_val, zeros(Float32, size(x_val)))
    z = flatten_last_two_dims(x)

    graph = topological_sort(z)
    forward!(graph)

    @test size(z.output) == (6, 4)
    @test isapprox(z.output, y_zyg; atol=1e-6)

    backward!(graph, ones(Float32, size(z.output)))
    @test isapprox(x.gradient, grad_zyg[1]; atol=1e-6)
end

@testset "Expression a * b + c" begin
    a_val, b_val, c_val = 2.0f0, 3.0f0, 5.0f0

    # Zygote reference
    f_zyg(a, b, c) = a * b + c
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val, c_val)
    grad_zyg = back_zyg(1.0f0)

    # MyAD graph
    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    c = Variable([c_val], [0.0f0])

    out = a*b + c

    graph = topological_sort(out)
    forward!(graph)
    @test isapprox(out.output[1], y_zyg; atol=1e-6)

    backward!(graph, [1.0f0])
    @test isapprox(a.gradient[1], grad_zyg[1]; atol=1e-6)
    @test isapprox(b.gradient[1], grad_zyg[2]; atol=1e-6)
    @test isapprox(c.gradient[1], grad_zyg[3]; atol=1e-6)
end

@testset "Scalar exponentiation (a ^ b)" begin
    a_val, b_val = 3.0f0, 2.0f0

    # Zygote ground truth
    f_zyg(a, b) = a ^ b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

    # MyAD
    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    out = a ^ b
    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output[1], y_zyg; atol=1e-6)

    backward!(graph, [1.0f0])
    @test isapprox(a.gradient[1], grad_zyg[1]; atol=1e-6)
    @test isapprox(b.gradient[1], grad_zyg[2]; atol=1e-6)
end

@testset "Scalar Expression a^2 + b" begin
    a_val, b_val = 3.0f0, 2.0f0

    # Zygote ground truth
    f_zyg(a, b) = a^2 + b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

    # MyAD
    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    out = a^2 + b
    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output[1], y_zyg; atol=1e-6)

    backward!(graph, [1.0f0])
    @test isapprox(a.gradient[1], grad_zyg[1]; atol=1e-6)
    @test isapprox(b.gradient[1], grad_zyg[2]; atol=1e-6)
end

@testset "Scalar Expression a^2 + b^2" begin
    a_val, b_val = 3.0f0, 2.0f0

    # Zygote ground truth
    f_zyg(a, b) = a^2 + b^2
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

    # MyAD
    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    out = a^2 + b^2
    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output[1], y_zyg; atol=1e-6)

    backward!(graph, [1.0f0])
    @test isapprox(a.gradient[1], grad_zyg[1]; atol=1e-6)
    @test isapprox(b.gradient[1], grad_zyg[2]; atol=1e-6)
end
