module MyAD

export GraphNode, Constant, Variable, ScalarOperator, MatMulOperator, BroadcastedOperator,
    forward!, backward!, topological_sort, relu, sigmoid, identity_fn,
    Conv1DOp, MaxPool1DOp, PermuteDimsOp, flatten_last_two_dims, flatten_last_two_dims_op

using Base.Threads: @threads
using LinearAlgebra: mul!

# === Abstract Node Type ===
abstract type GraphNode end

# === Basic Nodes ===
mutable struct Constant{T} <: GraphNode
    output::T
end

Constant(x::T) where T = Constant{T}(x)

mutable struct Variable{T} <: GraphNode
    output::AbstractArray{T}
    gradient::Union{Nothing,AbstractArray{T}}
end
Variable(data::AbstractArray{T}, grad::AbstractArray{T}) where {T} = Variable{T}(data, grad)

# === Operator Nodes ===
mutable struct ScalarOperator{F,T} <: GraphNode
    f::F
    inputs::Vector{GraphNode}
    output::Union{Nothing,AbstractArray{T}}
    gradient::Union{Nothing,AbstractArray{T}}
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
    output::Union{Nothing, AbstractArray{T}}
    gradient::Union{Nothing, AbstractArray{T}}
end

function flatten_last_two_dims(x::GraphNode)
    T = eltype(x.output)
    FlattenOp{T}(x, (), nothing, nothing)
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

function Conv1DOp(W::Variable{T}, b::Union{Variable{T}, Nothing}, input::GraphNode, kernel::Int, stride::Int, padding::Int, activation::Function) where T
    Conv1DOp{T}(W, b, input, kernel, stride, padding, activation,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

mutable struct MaxPool1DOp{T} <: GraphNode
    x::GraphNode
    kernel_size::Int
    stride::Int
    output::Union{Nothing, Array{T, 3}}   
    gradient::Union{Nothing, Array{T, 3}} 
    indices::Union{Nothing, Array{Int}}   
    dx::Union{Nothing, Array{T, 3}}    
end


function MaxPool1DOp(x::GraphNode, kernel_size::Int, stride::Int)
    T = eltype(x.output)
    MaxPool1DOp{T}(x, kernel_size, stride, nothing, nothing, nothing, nothing)
end

mutable struct PermuteDimsOp <: GraphNode
    x::GraphNode
    dims::NTuple{3, Int}
    output::Union{Nothing, AbstractArray}
    gradient::Union{Nothing, AbstractArray}
end

function PermuteDimsOp(x::GraphNode, dims::NTuple{3, Int})
    PermuteDimsOp(x, dims, nothing, nothing)
end



mutable struct EmbeddingOp{T} <: GraphNode
    weight::Variable{T}
    indices::Vector{Int}
    shape::Tuple
    output::Union{Nothing,AbstractArray{T}}
    gradient::Union{Nothing,AbstractArray{T}}
end

function EmbeddingOp(weight::Variable{T}, indices::Vector{Int}, shape::Tuple) where {T}
    EmbeddingOp{T}(weight, indices, shape, nothing, nothing)
end

# === Activation Functions ===
relu(x) = max.(zero(eltype(x)), x)
sigmoid(x) = one(eltype(x)) ./ (one(eltype(x)) .+ exp.(-x))
identity_fn(x) = x

# === Operator Overloading ===
import Base: +, *, -, /, sin, ^

Base.:+(a::GraphNode, b::GraphNode) = ScalarOperator(+, a, b)
Base.:*(a::GraphNode, b::GraphNode) = ScalarOperator(*, a, b)
Base.:-(a::GraphNode, b::GraphNode) = ScalarOperator(-, a, b)
Base.:/(a::GraphNode, b::GraphNode) = ScalarOperator(/, a, b)
Base.sin(x::GraphNode) = ScalarOperator(sin, x)
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
    node.output = node.f.(inputs...)
end

function forward(node::MatMulOperator)
    node.output = node.A.output * node.B.output
end

function forward(node::BroadcastedOperator)
    node.output = node.f.(node.input.output)
end

function forward(node::Constant) end
function forward(node::Variable) end

function forward(node::FlattenOp)
    node.orig_shape = size(node.x.output)
    node.output = reshape(node.x.output, :, size(node.x.output, ndims(node.x.output)))
end

function forward(node::Conv1DOp{T}) where {T}
    x = node.input.output             # shape (L, C, B)
    W = node.W.output                 # shape (K, C, O)
    b = node.b.output                 # shape (O, 1) или (O,)
    
    K, C, O = size(W)
    L, _, B = size(x)
    P = node.padding
    S = node.stride

    # === Padding ===
    x_padded = P > 0 ? cat(zeros(T, P, C, B), x, zeros(T, P, C, B); dims=1) : x
    node.x_padded = x_padded

    L_p = size(x_padded, 1)
    L_out = (L_p - K) ÷ S + 1

    # === im2col ===
    X_col = zeros(T, K * C, L_out * B)
    col = 1
    for b in 1:B
        for l in 0:L_out-1
            patch = view(x_padded, l*S+1:l*S+K, :, b)
            X_col[:, col] .= vec(patch)
            col += 1
        end
    end
    node.X_col = X_col

    # === Flip kernel along K ===
    W_flipped = W[K:-1:1, :, :]               # still (K, C, O)
    W_mat = reshape(W_flipped, K * C, O)
    node.W_mat = W_mat

    # === Convolution as matrix product ===
    out_mat = W_mat' * X_col                  # (O, L_out * B)

    # === Bias addition with safety check ===
    @assert size(b, 1) == O "Bias length must match number of output channels (O)"
    out_mat .+= reshape(b, O, 1)              # ensures broadcasting works

    node.out_mat = out_mat

    # === Reshape output to (L_out, O, B)
    out = reshape(out_mat, O, L_out, B)
    out = permutedims(out, (2, 1, 3))         # (O, L_out, B) → (L_out, O, B)

    # === Apply activation
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

# === Backward Pass ===
function accumulate_grad!(x::GraphNode, dx)
    if isnothing(x.gradient)
        x.gradient = similar(dx)
        x.gradient .= dx
    else
        @inbounds @simd for i in eachindex(x.gradient)
            x.gradient[i] += dx[i]
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
    elseif f == sin
        x = inputs[1]
        accumulate_grad!(x, out_grad .* cos.(x.output))
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
    accumulate_grad!(A, out_grad * B.output')
    accumulate_grad!(B, A.output' * out_grad)
end

function backward(node::BroadcastedOperator)
    f, x, out_grad = node.f, node.input, node.gradient
    δ = if f == relu
        out_grad .* (x.output .>= 0)
    elseif f == identity_fn
        out_grad
    elseif f == sigmoid
        σ = node.output
        out_grad .* σ .* (1 .- σ)
    elseif f == tanh
        out_grad .* (1 .- node.output .^ 2)
    else
        error("Unsupported broadcasted function $f")
    end
    accumulate_grad!(x, δ)
end

function backward(node::Constant) end
function backward(node::Variable) end

function backward(node::FlattenOp)
    grad = reshape(node.gradient, node.orig_shape)
    accumulate_grad!(node.x, grad)
end

function backward(node::Conv1DOp{T}) where {T}
    δ = node.gradient                          # shape (L_out, O, B)
    K, C, O = size(node.W.output)
    _, _, B = size(node.input.output)
    S, P = node.stride, node.padding
    L_out = size(δ, 1)

    # === Activation backward
    if node.activation == relu
        δ = δ .* (node.output .> 0)
    elseif node.activation == sigmoid
        σ = node.output
        δ = δ .* σ .* (1 .- σ)
    end

    # === Reshape δ to (O, L_out * B)
    dout_mat = permutedims(δ, (2, 1, 3))
    dout_mat = reshape(dout_mat, O, :)

    # === Gradient w.r.t. weights
    dW_mat = node.X_col * dout_mat'                   # (K⋅C, O)
    dW_flipped = reshape(dW_mat, K, C, O)             # (K, C, O)
    dW = dW_flipped[K:-1:1, :, :]                     # flip back to match W
    accumulate_grad!(node.W, dW)

    # === Gradient w.r.t. bias
    db = sum(dout_mat, dims=2)
    accumulate_grad!(node.b, reshape(db, size(node.b.output)))

    # === Gradient w.r.t. input
    dX_col = node.W_mat * dout_mat                    # (K*C, L_out * B)
    dx_pad = zeros(T, size(node.x_padded))
    col = 1
    for b in 1:B
        for l in 0:L_out-1
            patch = view(dx_pad, l*S+1:l*S+K, :, b)
            patch .+= reshape(dX_col[:, col], K, C)
            col += 1
        end
    end

    # === Unpad
    L_p = size(dx_pad, 1)
    dx = P > 0 ? view(dx_pad, P+1:L_p-P, :, :) : dx_pad
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
    for (i, idx) in enumerate(node.indices)
        acc[idx] = get!(acc, idx, zeros(eltype(dE_mat), size(dE_mat, 1)))
        @inbounds @views acc[idx] .+= dE_mat[:, i]
    end
    for (idx, val) in acc
        @inbounds @views node.weight.gradient[:, idx] .+= val
    end
end

function backward(node::GraphNode)
    error("No backward method defined for type $(typeof(node))")
end

end  # module
