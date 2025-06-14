using Test
include("../src/MyAD.jl")

using .MyAD
using Zygote
using Flux
using .MyAD: relu, sigmoid, identity_fn, tanh, flatten_last_two_dims, zero_grad!
using LinearAlgebra

@testset "Scalar addition (a + b)" begin
    a_val, b_val = 3.0f0, 2.0f0

    f_zyg(a, b) = a + b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    out = a + b
    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output[1], y_zyg; atol=1e-6)

    backward!(graph, [1.0f0])
    @test isapprox(a.gradient[1], grad_zyg[1]; atol=1e-6)
    @test isapprox(b.gradient[1], grad_zyg[2]; atol=1e-6)
end

@testset "Scalar multiplication (a * b)" begin
    a_val, b_val = 3.0f0, 2.0f0

    f_zyg(a, b) = a * b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    out = a * b
    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output[1], y_zyg; atol=1e-6)

    backward!(graph, [1.0f0])
    @test isapprox(a.gradient[1], grad_zyg[1]; atol=1e-6)
    @test isapprox(b.gradient[1], grad_zyg[2]; atol=1e-6)
end

@testset "Scalar division (a / b)" begin
    a_val, b_val = 6.0f0, 2.0f0

    f_zyg(a, b) = a / b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    out = a / b
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
    out = a - b
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

    f_zyg(A, B) = A * B
    y_zyg, back_zyg = Zygote.pullback(f_zyg, A_val, B_val)
    grad_zyg = back_zyg(ones(Float32, 2, 1))

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

    out = BroadcastedOperator(sigmoid, MatMulOperator(A, x) + b)

    graph = topological_sort(out)

    forward!(graph)
    @test isapprox(out.output, y_zyg; atol=1e-5)

    backward!(graph, ones(Float32, 4, 1))
    @test isapprox(A.gradient, grad_zyg[1]; atol=1e-5)
    @test isapprox(x.gradient, grad_zyg[2]; atol=1e-5)
    @test isapprox(b.gradient, grad_zyg[3]; atol=1e-5)
end

@testset "Flatten last two dims operation with batch" begin
    x_val = reshape(Float32.(1:24), (2, 3, 4))

    f_zyg(x) = reshape(x, :, size(x, ndims(x)))
    y_zyg, back_zyg = Zygote.pullback(f_zyg, x_val)
    grad_zyg = back_zyg(ones(Float32, 6, 4))

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

    f_zyg(a, b, c) = a * b + c
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val, c_val)
    grad_zyg = back_zyg(1.0f0)

    a = Variable([a_val], [0.0f0])
    b = Variable([b_val], [0.0f0])
    c = Variable([c_val], [0.0f0])

    out = a * b + c

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

    f_zyg(a, b) = a ^ b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

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

    f_zyg(a, b) = a^2 + b
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

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

    f_zyg(a, b) = a^2 + b^2
    y_zyg, back_zyg = Zygote.pullback(f_zyg, a_val, b_val)
    grad_zyg = back_zyg(1.0)

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

@testset "MaxPool1D operation compared with Flux/Zygote" begin
    x_val = reshape(Float32.(1:12), (6, 2, 1))
    x = Variable(x_val, zeros(Float32, size(x_val)))
    pool = MaxPool1DOp(x, 2, 2)

    graph = topological_sort(pool)
    forward!(graph)

    flux_out = maxpool(x_val, (2,), stride=(2,))
    @test isapprox(pool.output, flux_out; atol=1e-6)

    backward!(graph, ones(Float32, size(pool.output)))

    f_zyg(x) = maxpool(x, (2,), stride=(2,))
    _, back_zyg = Zygote.pullback(f_zyg, x_val)
    grad_zyg = back_zyg(ones(Float32, size(flux_out)))[1]

    @test isapprox(x.gradient, grad_zyg; atol=1e-6)
end

@testset "EmbeddingOp forward and backward" begin
    vocab_size = 10
    embedding_dim = 4
    sequence = [2 5 3; 1 4 2]

    weights = randn(Float32, embedding_dim, vocab_size)
    f_zyg(w) = reshape(w[:, vec(sequence)], (embedding_dim, size(sequence, 1), size(sequence, 2)))
    y_zyg, back_zyg = Zygote.pullback(f_zyg, weights)
    dy = ones(Float32, size(y_zyg))
    grad_zyg = back_zyg(dy)[1]

    W = Variable(copy(weights), zeros(Float32, size(weights)))
    embed = MyAD.EmbeddingOp(W, vec(sequence), (embedding_dim, size(sequence)...))

    graph = topological_sort(embed)
    forward!(graph)
    @test isapprox(embed.output, y_zyg; atol=1e-6)

    backward!(graph, dy)
    @test isapprox(W.gradient, grad_zyg; atol=1e-6)
end

@testset "PermuteDimsOp forward and backward" begin
    x_val = rand(Float32, 2, 3, 4)

    f_zyg(x) = permutedims(x, (3, 1, 2))
    y_zyg, back_zyg = Zygote.pullback(f_zyg, x_val)
    grad_zyg = back_zyg(ones(Float32, size(f_zyg(x_val))))[1]

    x = Variable(x_val, zeros(Float32, size(x_val)))
    z = PermuteDimsOp(x, (3, 1, 2))
    graph = topological_sort(z)

    forward!(graph)
    @test isapprox(z.output, y_zyg; atol=1e-6)

    backward!(graph, ones(Float32, size(y_zyg)))
    @test isapprox(x.gradient, grad_zyg; atol=1e-6)
end

@testset "Embedding -> Flatten -> Dense" begin
    vocab_size = 5
    embedding_dim = 3
    sequence = [1 2; 3 4]

    weights = randn(Float32, embedding_dim, vocab_size)
    dense_W = randn(Float32, 4, embedding_dim * size(sequence, 1))
    dense_b = randn(Float32, 4, 1)

    f_zyg(w_embed, w_dense, b_dense) = begin
        emb = reshape(w_embed[:, vec(sequence)], (embedding_dim, size(sequence)...))
        flat = reshape(emb, :, size(sequence, 2))
        sigmoid.(w_dense * flat .+ b_dense)
    end
    y_zyg, back_zyg = Zygote.pullback(f_zyg, weights, dense_W, dense_b)
    grad_zyg = back_zyg(ones(Float32, 4, size(sequence, 2)))

    embed = Variable(copy(weights), zeros(Float32, size(weights)))
    denseW = Variable(copy(dense_W), zeros(Float32, size(dense_W)))
    denseB = Variable(copy(dense_b), zeros(Float32, size(dense_b)))

    x = MyAD.EmbeddingOp(embed, vec(sequence), (embedding_dim, size(sequence)...))
    flat = flatten_last_two_dims(x)
    out = BroadcastedOperator(sigmoid, MatMulOperator(denseW, flat) + denseB)

    graph = topological_sort(out)
    forward!(graph)
    @test isapprox(out.output, y_zyg; atol=1e-5)

    zero_grad!(out)
    backward!(graph, ones(Float32, size(out.output)))

    # Сравнение градиентов
    function relative_error(a, b)
        return norm(a - b) / (norm(b) + eps(Float32))
    end

    used = unique(vec(sequence))

    for idx in used
        a = embed.gradient[:, idx]
        b = grad_zyg[1][:, idx]
        err = relative_error(a, b)

        @test err < 1e-4
    end

    @test isapprox(denseW.gradient, grad_zyg[2]; atol=1e-5)
    @test isapprox(denseB.gradient, grad_zyg[3]; atol=1e-5)
end



@testset "Conv1DOp correctness test" begin
    L, C, B = 10, 3, 2
    K, O = 3, 4

    x = rand(Float32, L, C, B)
    W = rand(Float32, K, C, O)
    b = rand(Float32, O)

    flux_conv = Flux.Conv((K,), C => O, identity; stride=1, pad=0)
    flux_conv.weight .= W
    flux_conv.bias .= b

    flux_output = flux_conv(x)

    x_var = Variable(x, zeros(Float32, size(x)))
    W_var = Variable(W, zeros(Float32, size(W)))
    b_var = Variable(reshape(b, O, 1), zeros(Float32, O, 1))

    myad_conv = Conv1DOp(W_var, b_var, x_var, K, 1, 0, identity_fn)
    nodes = topological_sort(myad_conv)
    forward!(nodes)
    myad_output = myad_conv.output

    @test isapprox(myad_output, flux_output; atol=1e-5)

    flux_loss(x, W, b) = sum(Flux.Conv((K,), C => O, identity; stride=1, pad=0, init=(o,i,kwargs...)->W, bias=b)(x))
    flux_grads = Zygote.gradient(flux_loss, x, W, b)

    backward!(nodes, ones(Float32, size(myad_output)))

    @test isapprox(x_var.gradient, flux_grads[1]; atol=1e-5)
    @test isapprox(W_var.gradient, flux_grads[2]; atol=1e-5)
    @test isapprox(vec(b_var.gradient), flux_grads[3]; atol=1e-5)

    println("Conv1DOp forward and backward gradients match Flux implementation!")
end