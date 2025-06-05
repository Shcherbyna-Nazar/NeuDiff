module MyAD

export GraphNode, Constant, Variable, ScalarOperator, MatMulOperator, BroadcastedOperator,
    forward!, backward!, topological_sort, relu, sigmoid, identity_fn, broadcast_add,
    Conv1DOp, MaxPool1DOp, PermuteDimsOp, flatten_last_two_dims, flatten_last_two_dims_op

# === Abstract Node Type ===
abstract type GraphNode end

# === Basic Nodes ===
mutable struct Constant{T} <: GraphNode
    output::T
    gradient::Union{Nothing,T}
end
Constant(x::T) where T = Constant{T}(x, nothing)

mutable struct Variable{T} <: GraphNode
    output::AbstractArray{T}
    gradient::AbstractArray{T}
end
Variable(x::AbstractArray{T}, g::AbstractArray{T}) where {T} = Variable{T}(x, g)

# === Operator Nodes ===
mutable struct ScalarOperator{F,T} <: GraphNode
    f::F
    inputs::Vector{GraphNode}
    output::Union{Nothing,T}
    gradient::Union{Nothing,T}
end
ScalarOperator(f::Function, args::GraphNode...) =
    ScalarOperator{typeof(f),eltype(args[1].output)}(f, collect(args), nothing, nothing)

mutable struct MatMulOperator{T} <: GraphNode
    A::GraphNode
    B::GraphNode
    output::Union{Nothing,AbstractMatrix{T}}
    gradient::Union{Nothing,AbstractMatrix{T}}
end
MatMulOperator(A::GraphNode, B::GraphNode) =
    MatMulOperator{eltype(A.output)}(A, B, nothing, nothing)

mutable struct BroadcastedOperator{F,T} <: GraphNode
    f::F
    input::GraphNode
    output::Union{Nothing,AbstractArray{T}}
    gradient::Union{Nothing,AbstractArray{T}}
end
BroadcastedOperator(f::Function, x::GraphNode) =
    BroadcastedOperator{typeof(f),eltype(x.output)}(f, x, nothing, nothing)

mutable struct FlattenOp{T} <: GraphNode
    x::GraphNode
    orig_shape::Tuple
    output::Union{Nothing,T}
    gradient::Union{Nothing,T}
end
function flatten_last_two_dims(x::GraphNode)
    T = eltype(x.output)
    return FlattenOp{T}(x, (), nothing, nothing)
end

mutable struct Conv1DOp{T} <: GraphNode
    W::Variable{T}
    b::Union{Variable{T},Nothing}
    input::GraphNode
    kernel::Int
    stride::Int
    padding::Int
    activation::Function
    output::Union{Nothing,AbstractArray{T}}
    gradient::Union{Nothing,AbstractArray{T}}
    X_col::Union{Nothing,AbstractArray{T}}
    x_padded::Union{Nothing,AbstractArray{T}}
    W_mat::Union{Nothing,AbstractArray{T}}
    out_mat::Union{Nothing,AbstractArray{T}}
    dx_padded::Union{Nothing,AbstractArray{T}}
    dX_col::Union{Nothing,AbstractArray{T}}
end
function Conv1D(in_channels, out_channels, kernel::Int, activation=identity;
    stride=1, padding=0)
    W = Variable(randn(out_channels, in_channels, kernel) * sqrt(2 / (in_channels * kernel)),
        zeros(out_channels, in_channels, kernel))
    b = Variable(zeros(out_channels, 1), zeros(out_channels, 1))
    return x -> Conv1DOp(W, b, x, kernel, stride, padding, activation,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

mutable struct MaxPool1DOp{T} <: GraphNode
    x::GraphNode
    kernel_size::Int
    stride::Int
    output::Union{Nothing,T}
    gradient::Union{Nothing,T}
    indices::Union{Nothing,Array{Int}}
    dx::Union{Nothing,T}
end
function MaxPool1DOp(x::GraphNode, kernel_size::Int, stride::Int,
    output::Union{Nothing,T}, gradient::Union{Nothing,T},
    indices::Union{Nothing,Array{Int}}) where {T}
    return MaxPool1DOp{T}(x, kernel_size, stride, output, gradient, indices)
end
function MaxPool1DOp(x::GraphNode, kernel_size::Int, stride::Int)
    T = eltype(x.output)
    return MaxPool1DOp{T}(x, kernel_size, stride, nothing, nothing, nothing, nothing)
end

mutable struct PermuteDimsOp{T} <: GraphNode
    x::GraphNode
    dims::NTuple{3,Int}
    output::Union{Nothing,T}
    gradient::Union{Nothing,T}
end
PermuteDimsOp(x::GraphNode, dims::NTuple{3,Int}) =
    PermuteDimsOp{eltype(x.output)}(x, dims, nothing, nothing)

mutable struct EmbeddingOp{T} <: GraphNode
    weight::Variable{T}
    indices::Vector{Int}
    shape::Tuple
    output::Union{Nothing,AbstractArray{T}}
    gradient::Union{Nothing,AbstractArray{T}}
end
function EmbeddingOp(weight::Variable{T}, indices::Vector{Int}, shape::Tuple) where {T}
    return EmbeddingOp{T}(weight, indices, shape, nothing, nothing)
end

# === Activation Functions ===
relu(x) = max.(0, x)
sigmoid(x) = 1.0 ./ (1.0 .+ exp.(-x))
identity_fn(x) = x
broadcast_add(a::AbstractMatrix, b::AbstractMatrix) = a .+ b * ones(size(b, 2))

# === Operator Overloading ===
import Base: +, *, -, /, sin
+(a::GraphNode, b::GraphNode) = ScalarOperator(+, a, b)
*(a::GraphNode, b::GraphNode) = ScalarOperator(*, a, b)
-(a::GraphNode, b::GraphNode) = ScalarOperator(-, a, b)
(/)(a::GraphNode, b::GraphNode) = ScalarOperator(/, a, b)
sin(x::GraphNode) = ScalarOperator(sin, x)

# === Topological Sort Utilities ===
function visit_children(::GraphNode, ::Set{GraphNode}, ::Vector{GraphNode})
    return nothing
end
function visit_children(node::ScalarOperator{F,T}, visited::Set{GraphNode}, order::Vector{GraphNode}) where {F,T}
    for x in node.inputs
        visit(x, visited, order)
    end
end
function visit_children(node::MatMulOperator{T}, visited::Set{GraphNode}, order::Vector{GraphNode}) where {T}
    visit(node.A, visited, order)
    visit(node.B, visited, order)
end
function visit_children(node::BroadcastedOperator{F,T}, visited::Set{GraphNode}, order::Vector{GraphNode}) where {F,T}
    visit(node.input, visited, order)
end
function visit_children(node::Conv1DOp{T}, visited::Set{GraphNode}, order::Vector{GraphNode}) where {T}
    visit(node.W, visited, order)
    if node.b !== nothing
        visit(node.b, visited, order)
    end
    visit(node.input, visited, order)
end
function visit_children(node::MaxPool1DOp{T}, visited::Set{GraphNode}, order::Vector{GraphNode}) where {T}
    visit(node.x, visited, order)
end
function visit_children(node::PermuteDimsOp{T}, visited::Set{GraphNode}, order::Vector{GraphNode}) where {T}
    visit(node.x, visited, order)
end
function visit_children(node::FlattenOp{T}, visited::Set{GraphNode}, order::Vector{GraphNode}) where {T}
    visit(node.x, visited, order)
end
function visit_children(node::EmbeddingOp{T}, visited::Set{GraphNode}, order::Vector{GraphNode}) where {T}
    visit(node.weight, visited, order)
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
    return order
end

# === Forward Pass ===
function forward!(nodes::Vector{GraphNode})
    for node in nodes
        forward(node)
    end
end

function forward(node::ScalarOperator)
    node.output = node.f(map(n -> n.output, node.inputs)...)
end
function forward(node::MatMulOperator)
    node.output = node.A.output * node.B.output
end
function forward(node::BroadcastedOperator)
    node.output = node.f.(node.input.output)
end
function forward(node::Constant) end
function forward(node::Variable) end
function forward(node::GraphNode)
    error("No forward method defined for type $(typeof(node))")
end

function forward(node::FlattenOp)
    node.orig_shape = size(node.x.output)
    node.output = reshape(node.x.output, :, size(node.x.output, ndims(node.x.output)))
end

using Base.Threads: @threads
using LinearAlgebra: mul!

function forward(node::Conv1DOp)
    @inbounds begin
        x = node.input.output
        W = node.W.output
        b = node.b === nothing ? nothing : node.b.output

        L, C, B = size(x)
        K = node.kernel
        S = node.stride
        P = node.padding

        L_p = L + 2P
        L_out = div(L_p - K, S) + 1
        O = size(W, 1)

        if node.x_padded === nothing || size(node.x_padded) != (L_p, C, B)
            node.x_padded = zeros(eltype(x), L_p, C, B)
        end
        @views node.x_padded[P+1:end-P, :, :] .= x
        x_padded = node.x_padded

        if node.X_col === nothing || size(node.X_col) != (C * K, L_out * B)
            node.X_col = zeros(eltype(x), C * K, L_out * B)
        end
        X_col = node.X_col

        if node.W_mat === nothing || size(node.W_mat) != (O, C * K)
            node.W_mat = reshape(W, O, :)
        end
        W_mat = node.W_mat

        if node.out_mat === nothing || size(node.out_mat) != (O, L_out * B)
            node.out_mat = zeros(eltype(x), O, L_out * B)
        end
        out_mat = node.out_mat

        @threads for bidx in 1:B
            col = 1
            for i in 1:S:(L_p-K+1)
                patch = @view x_padded[i:i+K-1, :, bidx]
                X_col[:, (bidx-1)*L_out+col] .= reshape(patch, :)
                col += 1
            end
        end

        mul!(out_mat, W_mat, X_col)

        if b !== nothing
            @threads for bidx in 1:B
                @views out_mat[:, (bidx-1)*L_out+1:bidx*L_out] .+= b
            end
        end

        out = reshape(out_mat, O, L_out, B)
        node.output = node.activation(permutedims(out, (2, 1, 3)))
    end
end

function forward(node::MaxPool1DOp)
    @inbounds begin
        x = node.x.output
        L, C, B = size(x)
        k, s = node.kernel_size, node.stride
        out_len = div(L - k, s) + 1

        if node.output === nothing || size(node.output) != (out_len, C, B)
            node.output = zeros(eltype(x), out_len, C, B)
        end
        if node.indices === nothing || size(node.indices) != (out_len, C, B)
            node.indices = zeros(Int, out_len, C, B)
        end

        out = node.output
        idx = node.indices

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
    node.output = permutedims(node.x.output, node.dims)
end

function forward(node::EmbeddingOp)
    node.output = reshape(node.weight.output[:, node.indices], node.shape)
end

# === Backward Pass ===
function accumulate_grad!(x::GraphNode, dx)
    if isnothing(x.gradient)
        x.gradient = dx
    else
        x.gradient .+= dx
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
        for input in inputs
            accumulate_grad!(input, out_grad)
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
    elseif f == sin
        x = inputs[1]
        accumulate_grad!(x, out_grad .* cos.(x.output))
    elseif f == broadcast_add
        a, b = inputs
        accumulate_grad!(a, out_grad)
        grad_b = sum(out_grad, dims=2)
        if !isnothing(b.gradient)
            b.gradient .+= convert.(eltype(b.gradient), grad_b)
        else
            b.gradient = convert.(eltype(b.output), grad_b)
        end
    elseif f == flatten_last_two_dims_op
        input = inputs[1]
        accumulate_grad!(input, reshape(out_grad, size(input.output)))
    else
        error("Unsupported scalar function $f")
    end
end

function backward(node::MatMulOperator)
    A, B, out_grad = node.A, node.B, node.gradient
    accumulate_grad!(A, out_grad * B.output')
    accumulate_grad!(B, A.output' * out_grad)
end

function backward(node::BroadcastedOperator)
    f, x, out_grad = node.f, node.input, node.gradient
    δ = if f == relu
        out_grad .* (x.output .> 0)
    elseif f == identity_fn
        out_grad
    elseif f == sigmoid
        σ = node.output
        out_grad .* σ .* (1 .- σ)
    else
        error("Unsupported broadcasted function $f")
    end
    x.gradient = isnothing(x.gradient) ? δ : x.gradient .+ δ
end

function backward(node::Constant) end
function backward(node::Variable) end
function backward(node::GraphNode)
    error("No backward method defined for type $(typeof(node))")
end

function backward(node::FlattenOp)
    grad = reshape(node.gradient, node.orig_shape)
    node.x.gradient = isnothing(node.x.gradient) ? grad : node.x.gradient .+ grad
end

function backward(node::Conv1DOp)
    @inbounds begin
        δy = node.gradient
        x = node.input.output
        W = node.W.output

        L, C, B = size(x)
        O, _, K = size(W)
        S = node.stride
        P = node.padding
        L_out = size(δy, 1)

        if node.activation == relu
            @. δy = δy * (node.output > 0)
        elseif node.activation == sigmoid
            σ = node.output
            @. δy = δy * σ * (1 - σ)
        elseif node.activation != identity_fn
            error("Unsupported activation function")
        end

        δy_mat = reshape(permutedims(δy, (2, 1, 3)), O, L_out * B)
        X_col = node.X_col
        dW_mat = δy_mat * X_col'
        node.W.gradient .= reshape(dW_mat, size(W))

        if node.b !== nothing
            db = sum(δy_mat, dims=2)
            node.b.gradient .= reshape(db, size(node.b.output))
        end

        W_mat = node.W_mat !== nothing ? node.W_mat : reshape(W, O, :)
        # ------------------------------
        # Preallocate dX_col if needed:
        dX_col_shape = (size(W_mat, 2), size(δy_mat, 2))
        if node.dX_col === nothing || size(node.dX_col) != dX_col_shape
            node.dX_col = zeros(eltype(W_mat), dX_col_shape...)
        else
            fill!(node.dX_col, 0)  # optional, mul! overwrites
        end
        # In-place matrix multiplication (no allocation!):
        mul!(node.dX_col, W_mat', δy_mat)
        dX_col = node.dX_col
        # ------------------------------

        if node.dx_padded === nothing || size(node.dx_padded) != (L + 2P, C, B)
            node.dx_padded = zeros(eltype(x), L + 2P, C, B)
        else
            fill!(node.dx_padded, 0)
        end
        dx_padded = node.dx_padded

        @threads for bidx in 1:B
            col = 1
            for i in 1:S:(L+2P-K+1)
                col_slice = @view dX_col[:, (bidx-1)*L_out+col]
                patch = reshape(col_slice, K, C)
                @views dx_padded[i:i+K-1, :, bidx] .+= patch
                col += 1
            end
        end

        dx = @view dx_padded[P+1:end-P, :, :]
        accumulate_grad!(node.input, dx)
    end
end


function backward(node::MaxPool1DOp)
    @inbounds begin
        x = node.x
        dy = node.gradient
        idx = node.indices
        L_out, C, B = size(dy)
        L, _, _ = size(x.output)

        if node.dx === nothing || size(node.dx) != size(x.output)
            node.dx = zeros(eltype(x.output), size(x.output))
        else
            fill!(node.dx, 0)
        end
        dx = node.dx

        @threads for b in 1:B
            for c in 1:C
                for i in 1:L_out
                    dx[idx[i, c, b], c, b] += dy[i, c, b]
                end
            end
        end

        if isnothing(x.gradient)
            x.gradient = copy(dx)
        else
            x.gradient .+= dx
        end
    end
end

function backward(node::PermuteDimsOp)
    rev = invperm(node.dims)
    grad = permutedims(node.gradient, rev)
    node.x.gradient = isnothing(node.x.gradient) ? grad : node.x.gradient .+ grad
end

function backward(node::EmbeddingOp)
    dE = node.gradient
    dE_mat = reshape(dE, size(dE, 1), :)
    acc = Dict{Int, Vector{eltype(dE_mat)}}()
    for (i, idx) in enumerate(node.indices)
        if !haskey(acc, idx)
            acc[idx] = zeros(eltype(dE_mat), size(dE_mat, 1))
        end
        @inbounds @views acc[idx] .+= dE_mat[:, i]
    end
    for (idx, val) in acc
        @inbounds @views node.weight.gradient[:, idx] .+= val
    end
end

end  # module
