module MyAD

export GraphNode, Constant, Variable, ScalarOperator, MatMulOperator, BroadcastedOperator,
    forward!, backward!, topological_sort, relu, sigmoid, identity_fn, broadcast_add, Conv1DOp, MaxPool1DOp, PermuteDimsOp,
    flatten_last_two_dims, flatten_last_two_dims_op, BatchNorm

# === Abstract Node Type ===
abstract type GraphNode end

# === Basic Nodes ===
mutable struct Constant{T} <: GraphNode
    output::T
    gradient::Any
end

Constant(x) = Constant(x, nothing)

mutable struct Variable <: GraphNode
    output::Any
    gradient::Any
end

# === Operator Nodes ===
mutable struct ScalarOperator{F} <: GraphNode
    f::F
    inputs::Vector{GraphNode}
    output::Any
    gradient::Any
end

mutable struct MatMulOperator <: GraphNode
    A::GraphNode
    B::GraphNode
    output::Any
    gradient::Any
end

mutable struct BroadcastedOperator{F} <: GraphNode
    f::F
    input::GraphNode
    output::Any
    gradient::Any
end

# === Activation Functions ===
relu(x) = max.(0, x)
sigmoid(x) = 1.0 ./ (1.0 .+ exp.(-x))
identity_fn(x) = x


function ScalarOperator(f::Function, args::GraphNode...)
    ScalarOperator{typeof(f)}(f, collect(args), nothing, nothing)
end

function MatMulOperator(A::GraphNode, B::GraphNode)
    MatMulOperator(A, B, nothing, nothing)
end

function BroadcastedOperator(f::Function, x::GraphNode)
    BroadcastedOperator{typeof(f)}(f, x, nothing, nothing)
end

# === Operator Overloading ===
import Base: +, *, -, /, sin
+(a::GraphNode, b::GraphNode) = ScalarOperator(+, a, b)
*(a::GraphNode, b::GraphNode) = ScalarOperator(*, a, b)
-(a::GraphNode, b::GraphNode) = ScalarOperator(-, a, b)
(/)(a::GraphNode, b::GraphNode) = ScalarOperator(/, a, b)
sin(x::GraphNode) = ScalarOperator(sin, x)

# === Topological Sort ===
function visit(node::GraphNode, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∉ visited
        push!(visited, node)
        if node isa ScalarOperator
            foreach(x -> visit(x, visited, order), node.inputs)
        elseif node isa MatMulOperator
            visit(node.A, visited, order)
            visit(node.B, visited, order)
        elseif node isa BroadcastedOperator
            visit(node.input, visited, order)
        elseif node isa Conv1DOp
            visit(node.x, visited, order)
        elseif node isa MaxPool1DOp
            visit(node.x, visited, order)
        end
        push!(order, node)
    end
end

function topological_sort(root::GraphNode)::Vector{GraphNode}
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit(root, visited, order)
    return order
end

# === Forward Pass ===
function forward(node::ScalarOperator)
    node.output = node.f(map(n -> n.output, node.inputs)...)
end

function forward(node::MatMulOperator)
    node.output = node.A.output * node.B.output
end

function forward(node::BroadcastedOperator)
    node.output = node.f.(node.input.output)
end

function forward(node::GraphNode)
    error("No forward method defined for node type $(typeof(node))")
end

function forward(node::Constant)
    nothing
end

function forward(node::Variable)
    nothing
end

using BenchmarkTools

# === Optimized forward pass ===
function forward!(nodes::Vector{GraphNode})
    for node in nodes
        forward(node)  # More precise and repeatable timing for each node
    end
end

broadcast_add(a::AbstractMatrix, b::AbstractMatrix) = a .+ b

# === Backward Pass ===
function backward(node::ScalarOperator)
    f, inputs, out_grad = node.f, node.inputs, node.gradient
    if f == +
        for input in inputs
            input.gradient .+= out_grad
        end
    elseif f == *
        a, b = inputs
        a.gradient .+= out_grad .* b.output
        b.gradient .+= out_grad .* a.output
    elseif f == - 
        a, b = inputs
        a.gradient .+= out_grad
        b.gradient .-= out_grad
    elseif f == /
        a, b = inputs
        a.gradient .+= out_grad ./ b.output
        b.gradient .-= out_grad .* a.output ./ (b.output .^ 2)
    elseif f == sin
        x = inputs[1]
        x.gradient .+= out_grad .* cos.(x.output)
    elseif f == broadcast_add
        a, b = inputs
        a.gradient .+= out_grad
        b.gradient .+= sum(out_grad, dims=2)
    elseif f == flatten_last_two_dims_op
        input = inputs[1]
        orig_shape = size(input.output)
        input.gradient .+= reshape(out_grad, orig_shape)

    else
        error("Unsupported function in ScalarOperator backward: $f")
    end
end

function backward(node::MatMulOperator)
    A, B, out_grad = node.A, node.B, node.gradient
    A.gradient .+= out_grad * B.output'
    B.gradient .+= A.output' * out_grad
end

function backward(node::BroadcastedOperator)
    f, x, out_grad = node.f, node.input, node.gradient
    if f == relu
        δ = out_grad .* (x.output .> 0)
    elseif f == identity_fn
        δ = out_grad
    elseif f == sigmoid
        σ = node.output
        δ = out_grad .* σ .* (1 .- σ)
    elseif f == tanh
        δ = out_grad .* (1 .- node.output .^ 2)
    else
        error("Unsupported function in BroadcastedOperator backward: $f")
    end

    x.gradient = isnothing(x.gradient) ? δ : x.gradient .+ δ
end

# === Optimized backward pass ===
function backward!(nodes::Vector{GraphNode}, seed=1.0)
    # Initialize gradients
    for node in nodes
        if !isnothing(node.output)
            node.gradient = zeros(size(node.output))
        end
    end
    last(nodes).gradient = seed

    # Measure the backward pass for each node
    for node in reverse(nodes)
        backward(node)  # More precise timing for each individual backward pass
    end
end

function backward(node::GraphNode)
    error("No backward method defined for node type $(typeof(node))")
end

function backward(node::Constant)
    nothing
end

function backward(node::Variable)
    nothing
end

mutable struct Conv1DOp <: GraphNode
    W::Variable                     # (out_ch, in_ch, kernel)
    b::Variable                     # (out_ch, 1)
    x::GraphNode                    # (in_ch, seq_len, batch)
    kernel_size::Int
    stride::Int
    output::Any
    gradient::Any
end

function Conv1DOp(W::Variable, b::Variable, x::GraphNode; stride=1)
    _, _, k = size(W.output)
    Conv1DOp(W, b, x, k, stride, nothing, nothing)
end

function im2col1d(x::AbstractArray{T, 3}, kernel_size::Int, stride::Int) where T
    in_ch, seq_len, batch = size(x)
    out_len = div(seq_len - kernel_size, stride) + 1
    col = Array{T}(undef, in_ch * kernel_size, out_len * batch)

    for b in 1:batch
        for i in 0:out_len - 1
            col_idx = b + i * batch
            patch = x[:, i*stride+1:i*stride+kernel_size, b]
            col[:, col_idx] .= reshape(patch, in_ch * kernel_size)
        end
    end
    return col, out_len
end

function col2im1d(dx_col::AbstractArray{T, 2}, in_ch::Int, seq_len::Int, kernel_size::Int, stride::Int, batch::Int) where T
    out_len = size(dx_col, 2) ÷ batch
    dx = zeros(T, in_ch, seq_len, batch)
    for b in 1:batch
        for i in 0:out_len - 1
            col_idx = b + i * batch
            patch = reshape(dx_col[:, col_idx], (in_ch, kernel_size))
            dx[:, i*stride+1:i*stride+kernel_size, b] .+= patch
        end
    end
    return dx
end

function forward(node::Conv1DOp)
    W, b, x = node.W.output, node.b.output, node.x.output
    out_ch, in_ch, k = size(W)
    in_ch2, seq_len, batch = size(x)
    @assert in_ch == in_ch2

    X_col, out_len = im2col1d(x, k, node.stride)
    W_mat = reshape(W, out_ch, in_ch * k)

    out = W_mat * X_col
    out .= out .+ b
    node.output = reshape(out, out_ch, out_len, batch)
end

function backward(node::Conv1DOp)
    W, b, x = node.W, node.b, node.x
    x_val, W_val = x.output, W.output
    dy = reshape(node.gradient, size(node.output, 1), :)  # (out_ch, out_len * batch)

    out_ch, in_ch, k = size(W_val)
    in_ch2, seq_len, batch = size(x_val)
    @assert in_ch == in_ch2

    X_col, out_len = im2col1d(x_val, k, node.stride)
    dW = dy * X_col'
    db = sum(dy, dims=2)
    dX_col = reshape(W_val, out_ch, in_ch * k)' * dy
    dx = col2im1d(dX_col, in_ch, seq_len, k, node.stride, batch)

    W.gradient = reshape(dW, size(W_val))
    b.gradient = db
    x.gradient = isnothing(x.gradient) ? dx : x.gradient .+ dx
end



# === Other Operations (Flatten, MaxPool1D, etc.) remain similar ===


export flatten_last_two_dims
function flatten_last_two_dims(x::GraphNode)
    return ScalarOperator(flatten_last_two_dims_op, x)
end

function flatten_last_two_dims_op(x::Array)
    return reshape(x, :, size(x, ndims(x)))
end

export flatten_last_two_dims_op


export MaxPool1DOp
mutable struct MaxPool1DOp <: GraphNode
    x::GraphNode  # вход: (channels, seq_len, batch)
    kernel_size::Int
    stride::Int
    output::Any
    gradient::Any
    indices::Any  # для сохранения позиций максимумов
end

function forward(node::MaxPool1DOp)
    
    x = node.x.output
    c, seq_len, b = size(x)
    k = node.kernel_size
    s = node.stride

    out_len = floor(Int, (seq_len - k) / s) + 1
    out = zeros(c, out_len, b)
    indices = Array{Int}(undef, c, out_len, b)

    @inbounds for batch in 1:b
        for ch in 1:c
            for i in 0:out_len-1
                idx_range = (i*s+1):(i*s+k)
                window = x[ch, idx_range, batch]
                max_val, max_idx = findmax(window)
                out[ch, i+1, batch] = max_val
                indices[ch, i+1, batch] = i*s + max_idx
            end
        end
    end


    node.output = out
    node.indices = indices
end

function backward(node::MaxPool1DOp)
    x = node.x
    dx = zeros(size(x.output))
    dy = node.gradient
    idx = node.indices

    c, out_len, b = size(dy)
    for batch in 1:b
        for ch in 1:c
            for i in 1:out_len
                max_pos = idx[ch, i, batch]
                dx[ch, max_pos, batch] += dy[ch, i, batch]
            end
        end
    end

    x.gradient = isnothing(x.gradient) ? dx : x.gradient .+ dx
end

export PermuteDimsOp

mutable struct PermuteDimsOp <: GraphNode
    x::GraphNode
    dims::NTuple{3, Int}
    output::Any
    gradient::Any
end

function PermuteDimsOp(x::GraphNode, dims::NTuple{3, Int})
    PermuteDimsOp(x, dims, nothing, nothing)
end

function forward(node::PermuteDimsOp)
    node.output = permutedims(node.x.output, node.dims)
end

function backward(node::PermuteDimsOp)
    node.x.gradient = isnothing(node.x.gradient) ? 
        permutedims(node.gradient, invperm(node.dims)) :
        node.x.gradient .+ permutedims(node.gradient, invperm(node.dims))
end




end # module
