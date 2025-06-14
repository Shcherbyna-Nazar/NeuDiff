using Base.Threads: @threads
using LinearAlgebra: mul!

# Backward pass for a list of nodes
function backward!(nodes::Vector{GraphNode}, seed=1.0)
    last(nodes).gradient = seed
    for node in reverse(nodes)
        backward(node)
    end
end

# Zero gradients for all variables in the graph
function zero_grad!(root::GraphNode)
    for node in topological_sort(root)
        if node isa Variable
            fill!(node.gradient, 0)
        end
    end
end

# Accumulate gradients for a node
function accumulate_grad!(x::GraphNode, dx)
    g = x.gradient
    if g === nothing || isempty(g)
        x.gradient = copy(dx)
    else
        @inbounds @simd for i in eachindex(g)
            g[i] += dx[i]
        end
    end
end

# Backward for scalar operators
function backward(node::ScalarOperator)
    f, inputs, out_grad = node.f, node.inputs, node.gradient
    if f == +
        a, b = inputs
        accumulate_grad!(a, out_grad)
        if size(b.output) != size(out_grad)
            grad_b = sum(out_grad, dims=2)
            accumulate_grad!(b, grad_b)
        else
            accumulate_grad!(b, out_grad)
        end
    elseif f == *
        a, b = inputs
        accumulate_grad!(a, out_grad .* b.output)
        accumulate_grad!(b, out_grad .* a.output)
    elseif f == -
        a, b = inputs
        accumulate_grad!(a, out_grad)
        accumulate_grad!(b, -out_grad)
    elseif f == /
        a, b = inputs
        accumulate_grad!(a, out_grad ./ b.output)
        accumulate_grad!(b, -out_grad .* a.output ./ (b.output .^ 2))
    elseif f == ^
        x, y = inputs
        if y isa Constant
            accumulate_grad!(x, out_grad .* y.output .* x.output .^ (y.output .- 1))
        else
            accumulate_grad!(x, out_grad .* y.output .* x.output .^ (y.output .- 1))
            accumulate_grad!(y, out_grad .* log.(x.output) .* x.output .^ y.output)
        end
    else
        error("Unsupported scalar function $f")
    end
end

# Backward for matrix multiplication
function backward(node::MatMulOperator)
    A, B, out_grad = node.A, node.B, node.gradient
    if A.gradient === nothing || size(A.gradient) != size(A.output)
        A.gradient = similar(A.output)
        fill!(A.gradient, 0)
    end
    if B.gradient === nothing || size(B.gradient) != size(B.output)
        B.gradient = similar(B.output)
        fill!(B.gradient, 0)
    end
    mul!(A.gradient, out_grad, B.output', 1.0, 1.0)
    mul!(B.gradient, A.output', out_grad, 1.0, 1.0)
end

# Backward for broadcasted operators (elementwise)
function backward(node::BroadcastedOperator)
    f, x, out_grad = node.f, node.input, node.gradient
    δ = x.gradient
    if δ === nothing || size(δ) != size(out_grad)
        δ = x.gradient = similar(out_grad)
        fill!(δ, 0)
    end

    if f === relu
        @inbounds @simd for i in eachindex(δ)
            δ[i] += x.output[i] ≥ 0 ? out_grad[i] : zero(eltype(out_grad))
        end
    elseif f === identity_fn
        @inbounds @simd for i in eachindex(δ)
            δ[i] += out_grad[i]
        end
    elseif f === sigmoid
        σ = node.output
        @inbounds @simd for i in eachindex(δ)
            δ[i] += out_grad[i] * σ[i] * (1 - σ[i])
        end
    elseif f === tanh
        σ = node.output
        @inbounds @simd for i in eachindex(δ)
            δ[i] += out_grad[i] * (1 - σ[i]^2)
        end
    else
        error("Unsupported broadcasted function $f")
    end
end

# No-op for constants and variables
backward(::Constant) = nothing
backward(::Variable) = nothing

# Backward for flatten operation
function backward(node::FlattenOp)
    grad = reshape(node.gradient, node.orig_shape)
    accumulate_grad!(node.x, grad)
end

# Backward for 1D convolution
function backward(node::Conv1DOp{T}) where {T}
    δ = node.gradient
    K, C, O = size(node.W.output)
    _, _, B = size(node.input.output)
    S, P = node.stride, node.padding
    L_out = size(δ, 1)

    # Activation backward
    if node.activation === relu
        δ = δ .* (node.output .> 0)
    elseif node.activation === sigmoid
        σ = node.output
        δ = δ .* σ .* (1 .- σ)
    end

    # Reshape δ to (O, L_out * B)
    dout_mat = reshape(permutedims(δ, (2, 1, 3)), O, :)

    # Gradient w.r.t. weights
    dW_mat = node.X_col * dout_mat'
    dW = reshape(dW_mat, K, C, O)[K:-1:1, :, :]
    accumulate_grad!(node.W, dW)

    # Bias grad
    if node.b !== nothing
        db = sum(dout_mat, dims=2)
        accumulate_grad!(node.b, reshape(db, size(node.b.output)))
    end

    # dX_col
    dX_col = node.dX_col
    if dX_col === nothing || size(dX_col) != (K*C, L_out*B)
        dX_col = node.dX_col = similar(node.X_col)
    end
    mul!(dX_col, node.W_mat, dout_mat)

    # dx_padded
    dx_pad = node.dx_padded
    x_pad_shape = size(node.x_padded)
    if dx_pad === nothing || size(dx_pad) != x_pad_shape
        dx_pad = node.dx_padded = zeros(T, x_pad_shape)
    else
        fill!(dx_pad, 0)
    end

    col = 1
    @inbounds @simd for b in 1:B
        for l in 0:L_out-1
            patch = @view dx_pad[l*S+1:l*S+K, :, b]
            @views patch .+= reshape(dX_col[:, col], K, C)
            col += 1
        end
    end

    # Unpad
    if P > 0
        dx = @view dx_pad[P+1:P+size(node.input.output, 1)+P, :, :]
    else
        dx = dx_pad
    end
    accumulate_grad!(node.input, dx)
end

# Backward for max pooling
function backward(node::MaxPool1DOp)
    x, dy, idx = node.x, node.gradient, node.indices
    L_out, C, B = size(dy)
    node.dx = node.dx !== nothing && size(node.dx) == size(x.output) ? (fill!(node.dx, 0); node.dx) : zeros(eltype(x.output), size(x.output))
    dx = node.dx

    @threads for b in 1:B
        for c in 1:C
            for i in 1:L_out
                dx[idx[i, c, b], c, b] += dy[i, c, b]
            end
        end
    end

    accumulate_grad!(x, dx)
end

# Backward for permutedims
function backward(node::PermuteDimsOp)
    rev = invperm(node.dims)
    grad = permutedims(node.gradient, rev)
    accumulate_grad!(node.x, grad)
end

# Backward for embedding
function backward(node::EmbeddingOp)
    dE = node.gradient
    dE_mat = reshape(dE, size(dE, 1), :)
    grad = node.weight.gradient
    indices = node.indices

    # Zero only used indices
    for idx in unique(indices)
        @views grad[:, idx] .= 0
    end

    # Accumulate gradients
    for (col_idx, word_idx) in enumerate(indices)
        @views grad[:, word_idx] .+= dE_mat[:, col_idx]
    end
end

# Fallback for unsupported nodes
function backward(node::GraphNode)
    error("No backward method defined for type $(typeof(node))")
end
