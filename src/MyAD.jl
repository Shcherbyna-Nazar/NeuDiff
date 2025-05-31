
module MyAD

export GraphNode, Constant, Variable, ScalarOperator, MatMulOperator, BroadcastedOperator,
       forward!, backward!, topological_sort, relu, sigmoid, identity_fn, broadcast_add,
       Conv1DOp, MaxPool1DOp, PermuteDimsOp, flatten_last_two_dims, flatten_last_two_dims_op

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

# === Scalar Operator ===
mutable struct ScalarOperator{F} <: GraphNode
    f::F
    inputs::Vector{GraphNode}
    output::Any
    gradient::Any
end

ScalarOperator(f::Function, args::GraphNode...) = ScalarOperator{typeof(f)}(f, collect(args), nothing, nothing)

# === Matrix Multiplication Operator ===
mutable struct MatMulOperator <: GraphNode
    A::GraphNode
    B::GraphNode
    output::Any
    gradient::Any
end

MatMulOperator(A::GraphNode, B::GraphNode) = MatMulOperator(A, B, nothing, nothing)

# === Broadcasted Operator ===
mutable struct BroadcastedOperator{F} <: GraphNode
    f::F
    input::GraphNode
    output::Any
    gradient::Any
end

BroadcastedOperator(f::Function, x::GraphNode) = BroadcastedOperator{typeof(f)}(f, x, nothing, nothing)

# === Activation Functions ===
relu(x) = max.(0, x)
sigmoid(x) = 1.0 ./ (1.0 .+ exp.(-x))
identity_fn(x) = x
broadcast_add(a::AbstractMatrix, b::AbstractMatrix) = a .+ b

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
        elseif node isa PermuteDimsOp
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

# === Forward ===
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

# === Backward ===
function backward!(nodes::Vector{GraphNode}, seed=1.0)
    for node in nodes
        if !isnothing(node.output)
            node.gradient = zeros(size(node.output))
        end
    end
    last(nodes).gradient = seed
    for node in reverse(nodes)
        backward(node)
    end
end

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
        input.gradient .+= reshape(out_grad, size(input.output))
    else
        error("Unsupported scalar function $f")
    end
end

function backward(node::MatMulOperator)
    A, B, out_grad = node.A, node.B, node.gradient
    A.gradient .+= out_grad * B.output'
    B.gradient .+= A.output' * out_grad
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

# === Flatten
flatten_last_two_dims(x::GraphNode) = ScalarOperator(flatten_last_two_dims_op, x)
flatten_last_two_dims_op(x::Array) = reshape(x, :, size(x, ndims(x)))

# === Conv1D Operator
mutable struct Conv1DOp <: GraphNode
    W::Variable
    b::Variable
    x::GraphNode
    kernel_size::Int
    stride::Int
    output::Any
    gradient::Any
end

Conv1DOp(W::Variable, b::Variable, x::GraphNode; stride=1) = Conv1DOp(W, b, x, size(W.output, 3), stride, nothing, nothing)

function im2col1d(x::AbstractArray{T, 3}, k::Int, s::Int) where T
    in_ch, seq_len, batch = size(x)
    out_len = div(seq_len - k, s) + 1
    col = Matrix{T}(undef, in_ch * k, out_len * batch)
    @inbounds for b in 1:batch, i in 0:out_len-1
        col[:, b + i*batch] .= vec(x[:, i*s+1:i*s+k, b])
    end
    return col, out_len
end

function col2im1d(dx_col::AbstractArray{T, 2}, in_ch::Int, seq_len::Int, k::Int, s::Int, batch::Int) where T
    out_len = size(dx_col, 2) ÷ batch
    dx = zeros(T, in_ch, seq_len, batch)
    @inbounds for b in 1:batch, i in 0:out_len-1
        dx[:, i*s+1:i*s+k, b] .+= reshape(dx_col[:, b + i*batch], in_ch, k)
    end
    return dx
end

function forward(node::Conv1DOp)
    W, b, x = node.W.output, node.b.output, node.x.output
    out_ch, in_ch, k = size(W)
    X_col, out_len = im2col1d(x, k, node.stride)
    W_mat = reshape(W, out_ch, in_ch * k)
    out = W_mat * X_col .+ b
    node.output = reshape(out, out_ch, out_len, size(x, 3))
end

function backward(node::Conv1DOp)
    W, b, x = node.W, node.b, node.x
    dy = reshape(node.gradient, size(node.output, 1), :)
    X_col, _ = im2col1d(x.output, node.kernel_size, node.stride)
    dW = dy * X_col'
    db = sum(dy, dims=2)
    dX_col = reshape(W.output, size(W.output,1), :)' * dy
    dx = col2im1d(dX_col, size(x.output,1), size(x.output,2), node.kernel_size, node.stride, size(x.output,3))
    W.gradient = reshape(dW, size(W.output))
    b.gradient = db
    x.gradient = isnothing(x.gradient) ? dx : x.gradient .+ dx
end

# === MaxPool1D
mutable struct MaxPool1DOp <: GraphNode
    x::GraphNode
    kernel_size::Int
    stride::Int
    output::Any
    gradient::Any
    indices::Any
end

function forward(node::MaxPool1DOp)
    x = node.x.output
    c, seq_len, b = size(x)
    k, s = node.kernel_size, node.stride
    out_len = (seq_len - k) ÷ s + 1
    out = zeros(c, out_len, b)
    idx = similar(out, Int)
    @inbounds for batch in 1:b, ch in 1:c, i in 0:out_len-1
        r = i*s+1:i*s+k
        win = x[ch, r, batch]
        max_val, max_idx = findmax(win)
        out[ch, i+1, batch] = max_val
        idx[ch, i+1, batch] = r.start + max_idx - 1
    end
    node.output, node.indices = out, idx
end

function backward(node::MaxPool1DOp)
    x = node.x
    dx = zeros(size(x.output))
    dy, idx = node.gradient, node.indices
    c, out_len, b = size(dy)
    @inbounds for batch in 1:b, ch in 1:c, i in 1:out_len
        dx[ch, idx[ch,i,batch], batch] += dy[ch,i,batch]
    end
    x.gradient = isnothing(x.gradient) ? dx : x.gradient .+ dx
end

# === PermuteDimsOp
mutable struct PermuteDimsOp <: GraphNode
    x::GraphNode
    dims::NTuple{3, Int}
    output::Any
    gradient::Any
end

PermuteDimsOp(x::GraphNode, dims::NTuple{3, Int}) = PermuteDimsOp(x, dims, nothing, nothing)

function forward(node::PermuteDimsOp)
    node.output = permutedims(node.x.output, node.dims)
end

function backward(node::PermuteDimsOp)
    rev = invperm(node.dims)
    grad = permutedims(node.gradient, rev)
    node.x.gradient = isnothing(node.x.gradient) ? grad : node.x.gradient .+ grad
end

end  # module
