module MyAD

export GraphNode, Constant, Variable, ScalarOperator, MatMulOperator, BroadcastedOperator,
    forward!, backward!, topological_sort, relu, sigmoid, identity_fn,
    Conv1DOp, MaxPool1DOp, PermuteDimsOp, flatten_last_two_dims, flatten_last_two_dims_op

using Base.Threads: @threads
using LinearAlgebra: mul!

# === Abstract Node Type ===
abstract type GraphNode end

# === Constants ===
mutable struct Constant{T} <: GraphNode
    output::T
end
Constant(x::T) where {T} = Constant{T}(x)

# === Variables (with gradient) ===
mutable struct Variable{T, N} <: GraphNode
    output::Array{T, N}
    gradient::Array{T, N}
end
Variable(data::Array{T, N}) where {T, N} = Variable{T, N}(data, zeros(T, size(data)))
Variable(data::AbstractArray{T, N}) where {T, N} =
    Variable{T, N}(Array(data), zeros(T, size(data)))
Variable(data::AbstractArray{T, N}, grad::AbstractArray{T, N}) where {T, N} =
    Variable{T, N}(Array(data), Array(grad))


# === Operator Nodes ===
mutable struct ScalarOperator{F, T, N} <: GraphNode
    f::F
    inputs::NTuple{2, GraphNode}
    output::Array{T, N}
    gradient::Array{T, N}
end

function ScalarOperator(f::F, a::GraphNode, b::GraphNode) where {F}
    T = promote_type(eltype(a.output), eltype(b.output))
    N = max(ndims(a.output), ndims(b.output))
    # Always allocate an empty array of N dimensions:
    empty_shape = ntuple(_ -> 0, N)
    ScalarOperator{F, T, N}(f, (a, b), Array{T}(undef, empty_shape...), Array{T}(undef, empty_shape...))
end



mutable struct MatMulOperator{T, NA, NB} <: GraphNode
    A::GraphNode
    B::GraphNode
    output::Array{T, 2}
    gradient::Array{T, 2}
end

function MatMulOperator(A::GraphNode, B::GraphNode)
    T = promote_type(eltype(A.output), eltype(B.output))
    MatMulOperator{T, ndims(A.output), ndims(B.output)}(A, B, Array{T}(undef, 0, 0), Array{T}(undef, 0, 0))
end

mutable struct BroadcastedOperator{F, T, N} <: GraphNode
    f::F
    input::GraphNode
    output::Array{T, N}
    gradient::Array{T, N}
end

function BroadcastedOperator(f::F, x::GraphNode) where {F}
    T = eltype(x.output)
    shape = size(x.output)
    BroadcastedOperator{F, T, length(shape)}(f, x, zeros(T, shape), zeros(T, shape))
end

mutable struct FlattenOp{T, N, NO} <: GraphNode
    x::GraphNode
    orig_shape::NTuple{N, Int}
    output::Array{T, NO}
    gradient::Array{T, NO}
end

function flatten_last_two_dims(x::GraphNode)
    T = eltype(x.output)
    orig_shape = size(x.output)
    out_shape = (:, size(x.output, ndims(x.output)))
    FlattenOp{T, length(orig_shape), 2}(x, orig_shape, Array{T}(undef, 0, 0), Array{T}(undef, 0, 0))
end

mutable struct Conv1DOp{T, F, NW, NB, NI} <: GraphNode
    W::Variable{T, NW}
    b::Union{Variable{T, NB}, Nothing}
    input::GraphNode
    kernel::Int
    stride::Int
    padding::Int
    activation::F
    output::Array{T, 3}
    gradient::Array{T, 3}
    X_col::Array{T, 2}
    x_padded::Array{T, 3}
    W_mat::Array{T, 2}
    out_mat::Array{T, 2}
    dx_padded::Array{T, 3}
    dX_col::Array{T, 2}
    W_mat_T::Array{T, 2}
end

function Conv1DOp(W::Variable{T, NW}, b::Union{Variable{T, NB}, Nothing}, input::GraphNode,
                  kernel::Int, stride::Int, padding::Int, activation::F) where {T, NW, NB, F}
    Conv1DOp{T, F, NW, NB, ndims(input.output)}(
        W, b, input, kernel, stride, padding, activation,
        Array{T, 3}(undef, 0, 0, 0), Array{T, 3}(undef, 0, 0, 0),
        Array{T, 2}(undef, 0, 0), Array{T, 3}(undef, 0, 0, 0),
        Array{T, 2}(undef, 0, 0), Array{T, 2}(undef, 0, 0),
        Array{T, 3}(undef, 0, 0, 0), Array{T, 2}(undef, 0, 0),
        Array{T, 2}(undef, 0, 0)
    )
end

mutable struct MaxPool1DOp{T} <: GraphNode
    x::GraphNode
    kernel_size::Int
    stride::Int
    output::Array{T, 3}
    gradient::Array{T, 3}
    indices::Array{Int, 3}
    dx::Array{T, 3}
end

function MaxPool1DOp(x::GraphNode, kernel_size::Int, stride::Int)
    T = eltype(x.output)
    MaxPool1DOp{T}(x, kernel_size, stride,
        Array{T, 3}(undef, 0, 0, 0), Array{T, 3}(undef, 0, 0, 0),
        Array{Int, 3}(undef, 0, 0, 0), Array{T, 3}(undef, 0, 0, 0))
end

mutable struct PermuteDimsOp{T, N} <: GraphNode
    x::GraphNode
    dims::NTuple{N, Int}
    output::Array{T, N}
    gradient::Array{T, N}
end

function PermuteDimsOp(x::GraphNode, dims::NTuple{N, Int}) where {N}
    T = eltype(x.output)
    PermuteDimsOp{T, N}(x, dims, Array{T, N}(undef, 0, 0, 0), Array{T, N}(undef, 0, 0, 0))
end

mutable struct EmbeddingOp{T, N} <: GraphNode
    weight::Variable{T, 2}
    indices::Vector{Int}
    shape::NTuple{N, Int}
    output::Array{T, N}
    gradient::Array{T, N}
end

function EmbeddingOp(weight::Variable{T, 2}, indices::Vector{Int}, shape::NTuple{N, Int}) where {T, N}
    EmbeddingOp{T, N}(weight, indices, shape, Array{T, N}(undef, shape...), Array{T, N}(undef, shape...))
end


# === Activation Functions ===
relu(x) = max.(zero(eltype(x)), x)
sigmoid(x) = one(eltype(x)) ./ (one(eltype(x)) .+ exp.(-x))
identity_fn(x) = x

# === Operator Overloading ===
import Base: +, *, -, /, ^

Base.:+(a::GraphNode, b::GraphNode) = ScalarOperator(+, a, b)
Base.:*(a::GraphNode, b::GraphNode) = ScalarOperator(*, a, b)
Base.:-(a::GraphNode, b::GraphNode) = ScalarOperator(-, a, b)
Base.:/(a::GraphNode, b::GraphNode) = ScalarOperator(/, a, b)
Base.:^(a::GraphNode, b::GraphNode) = ScalarOperator(^, a, b)
Base.:^(a::GraphNode, b::Number) = ScalarOperator(^, a, Constant(b))

# === Topological Sort Utilities ===
function visit_children(node::ScalarOperator, visited::Set{GraphNode}, order::Vector{GraphNode})
    for x in node.inputs
        visit(x, visited, order)
    end
end

function visit_children(node::MatMulOperator, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.A, visited, order)
    visit(node.B, visited, order)
end

function visit_children(node::BroadcastedOperator, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.input, visited, order)
end

function visit_children(node::Conv1DOp, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.W, visited, order)
    if node.b !== nothing
        visit(node.b, visited, order)
    end
    visit(node.input, visited, order)
end

function visit_children(node::MaxPool1DOp, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.x, visited, order)
end

function visit_children(node::PermuteDimsOp, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.x, visited, order)
end

function visit_children(node::FlattenOp, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.x, visited, order)
end

function visit_children(node::EmbeddingOp, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.weight, visited, order)
end

function visit_children(::GraphNode, ::Set{GraphNode}, ::Vector{GraphNode})
    # fallback for unsupported GraphNode types: no children
end

function visit(node::GraphNode, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∉ visited
        push!(visited, node)
        visit_children(node, visited, order)
        push!(order, node)
    end
end


function topological_sort(root::GraphNode)
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit(root, visited, order)
    order
end

# === Forward Pass ===
function forward!(nodes::Vector{GraphNode})
    for node in nodes
        forward(node)
    end
end

function forward(node::ScalarOperator)
    inputs = map(n -> n.output, node.inputs)
    if node.output === nothing || size(node.output) != size(inputs[1])
        node.output = similar(inputs[1])
    end
    broadcast!(node.f, node.output, inputs[1], inputs[2])
end

function forward(node::MatMulOperator)
    A, B = node.A.output, node.B.output
    if node.output === nothing || size(node.output) != (size(A,1), size(B,2))
        node.output = similar(A, size(A,1), size(B,2))
    end
    mul!(node.output, A, B)
end

function forward(node::BroadcastedOperator)
    x = node.input.output
    if node.output === nothing || size(node.output) != size(x)
        node.output = similar(x)
    end
    broadcast!(node.f, node.output, x)
end

function forward(node::Constant) end
function forward(node::Variable) end

function forward(node::FlattenOp)
    node.orig_shape = size(node.x.output)
    out_shape = (:, size(node.x.output, ndims(node.x.output)))
    if node.output === nothing || size(node.output) != (prod(node.orig_shape[1:end-1]), node.orig_shape[end])
        node.output = reshape(node.x.output, prod(node.orig_shape[1:end-1]), node.orig_shape[end])
    else
        reshape!(node.output, node.x.output, prod(node.orig_shape[1:end-1]), node.orig_shape[end])
    end
end

function forward(node::Conv1DOp{T}) where {T}
    x = node.input.output
    W = node.W.output
    b = node.b === nothing ? nothing : node.b.output

    K, C, O = size(W)
    L, _, B = size(x)
    P, S = node.padding, node.stride
    L_padded = L + 2P
    L_out = (L_padded - K) ÷ S + 1

    # === Allocate or reuse padded input ===
    if node.x_padded === nothing || size(node.x_padded) != (L_padded, C, B)
        node.x_padded = zeros(T, L_padded, C, B)
    else
        fill!(node.x_padded, 0)
    end
    @views node.x_padded[P+1:P+L, :, :] .= x
    x_padded = node.x_padded

    # === im2col transform ===
    if node.X_col === nothing || size(node.X_col) != (K * C, L_out * B)
        node.X_col = zeros(T, K * C, L_out * B)
    end
    X_col = node.X_col

    col = 1
    @inbounds for b in 1:B
        for l in 0:L_out-1
            patch = @view x_padded[l*S+1:l*S+K, :, b]
            @views X_col[:, col] .= vec(patch)
            col += 1
        end
    end

    # === Flip and reshape kernel ===
    W_flipped = @view W[K:-1:1, :, :]
    if node.W_mat === nothing || size(node.W_mat) != (K*C, O)
        node.W_mat = reshape(copy(W_flipped), K*C, O)
        node.W_mat_T = transpose(node.W_mat)
    else
        reshape!(node.W_mat, copy(W_flipped), K*C, O)
        node.W_mat_T = transpose(node.W_mat)
    end
    W_mat_T = node.W_mat_T  # (O, K*C)

    # === Matrix multiplication ===
    if node.out_mat === nothing || size(node.out_mat) != (O, L_out * B)
        node.out_mat = similar(W_mat_T, O, L_out * B)
    end
    out_mat = node.out_mat

    mul!(out_mat, W_mat_T, X_col)  # O × (K*C) × (K*C × L_out*B)

    # === Bias addition
    if b !== nothing
        @inbounds @views out_mat .+= reshape(b, O, 1)
    end

    # === Reshape and activate ===
    out = reshape(out_mat, O, L_out, B)
    out = permutedims(out, (2, 1, 3))  # (L_out, O, B)
    node.output = node.activation === identity_fn ? out : node.activation.(out)
end


function forward(node::MaxPool1DOp)
    @inbounds begin
        x = node.x.output
        L, C, B = size(x)
        k, s = node.kernel_size, node.stride
        out_len = div(L - k, s) + 1

        node.output = node.output !== nothing && size(node.output) == (out_len, C, B) ? node.output : zeros(eltype(x), out_len, C, B)
        node.indices = node.indices !== nothing && size(node.indices) == (out_len, C, B) ? node.indices : zeros(Int, out_len, C, B)
        out, idx = node.output, node.indices

        @threads for b in 1:B
            for c in 1:C
                for i in 0:out_len-1
                    r = i*s+1:i*s+k
                    win = @view x[r, c, b]
                    max_val, max_idx = findmax(win)
                    out[i+1, c, b] = max_val
                    idx[i+1, c, b] = r.start + max_idx - 1
                end
            end
        end
    end
end

function forward(node::PermuteDimsOp)
    x = node.x.output
    T = eltype(x)  # <- теперь x.output уже определён
    out = permutedims(x, node.dims)
    
    result = Array{T, 3}(undef, size(out)...)
    copyto!(result, out)
    node.output = result
end

function forward(node::EmbeddingOp)
    node.output = reshape(node.weight.output[:, node.indices], node.shape)
end

function forward(node::GraphNode)
    error("No forward method defined for type $(typeof(node))")
end

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


function backward!(nodes::Vector{GraphNode}, seed=1.0)
    last(nodes).gradient = seed
    for node in reverse(nodes)
        backward(node)
    end
end

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

function backward(node::BroadcastedOperator)
    f = node.f
    x = node.input
    out_grad = node.gradient
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


function backward(node::Constant) end
function backward(node::Variable) end

function backward(node::FlattenOp)
    grad = reshape(node.gradient, node.orig_shape)
    accumulate_grad!(node.x, grad)
end

function backward(node::Conv1DOp{T}) where {T}
    δ = node.gradient
    K, C, O = size(node.W.output)
    _, _, B = size(node.input.output)
    S, P = node.stride, node.padding
    L_out = size(δ, 1)

    # === Activation backward
    if node.activation === relu
        δ = δ .* (node.output .> 0)
    elseif node.activation === sigmoid
        σ = node.output
        δ = δ .* σ .* (1 .- σ)
    end

    # === Reshape δ to (O, L_out * B)
    dout_mat = reshape(permutedims(δ, (2, 1, 3)), O, :)

    # === Gradient w.r.t. weights
    dW_mat = node.X_col * dout_mat'
    dW = reshape(dW_mat, K, C, O)[K:-1:1, :, :]
    accumulate_grad!(node.W, dW)

    # === Bias grad
    if node.b !== nothing
        db = sum(dout_mat, dims=2)
        accumulate_grad!(node.b, reshape(db, size(node.b.output)))
    end

    # === dX_col
    dX_col = node.dX_col
    if dX_col === nothing || size(dX_col) != (K*C, L_out*B)
        dX_col = node.dX_col = similar(node.X_col)
    end
    mul!(dX_col, node.W_mat, dout_mat)

    # === dx_padded
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

    # === Unpad
    if P > 0
        dx = @view dx_pad[P+1:P+size(node.input.output, 1)+P, :, :]
    else
        dx = dx_pad
    end
    accumulate_grad!(node.input, dx)
end

function backward(node::MaxPool1DOp)
    @inbounds begin
        x, dy, idx = node.x, node.gradient, node.indices
        L_out, C, B = size(dy)
        L, _, _ = size(x.output)

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
end

function backward(node::PermuteDimsOp)
    rev = invperm(node.dims)
    grad = permutedims(node.gradient, rev)
    accumulate_grad!(node.x, grad)
end

function backward(node::EmbeddingOp)
    dE = node.gradient
    dE_mat = reshape(dE, size(dE, 1), :)
    acc = Dict{Int, Vector{eltype(dE_mat)}}()
    used = Set(node.indices)
    for idx in used
        fill!(node.weight.gradient[:, idx], 0)
    end
    for (i, idx) in enumerate(node.indices)
        acc[idx] = get!(acc, idx, zeros(eltype(dE_mat), size(dE_mat, 1)))
        acc[idx] .+= dE_mat[:, i]
    end
    for (idx, val) in acc
        node.weight.gradient[:, idx] .+= val
    end
end


function backward(node::GraphNode)
    error("No backward method defined for type $(typeof(node))")
end

function zero_grad!(root::GraphNode)
    for node in topological_sort(root)
        if node isa Variable
            fill!(node.gradient, 0)
        end
    end
end


end  # module
