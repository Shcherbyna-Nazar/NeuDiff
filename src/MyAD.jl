module MyAD

export GraphNode, Constant, Variable, ScalarOperator, MatMulOperator, BroadcastedOperator,
    forward!, backward!, topological_sort, relu, sigmoid, identity_fn, broadcast_add

# === Abstract Node Type ===
abstract type GraphNode end

# === Basic Nodes ===
struct Constant{T} <: GraphNode
    output::T
end

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
    # Constants do not change, so no need to compute output
    nothing
end

function forward(node::Variable)
    # Variables are inputs, so we just return their output
    nothing
end

function forward!(nodes::Vector{GraphNode})
    for node in nodes
        forward(node)
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


function backward!(nodes::Vector{GraphNode}, seed=1.0)
    for node in nodes
        if !isnothing(node.output)
            node.gradient = zeros(size(node.output))
        end
    end
    last(nodes).gradient = seed
    for node in reverse(nodes)
        backward(node)  # no type checks
    end
end


function backward(node::GraphNode)
    error("No backward method defined for node type $(typeof(node))")
end

function backward(node::Constant)
    # Constants do not have gradients
    nothing
end

function backward(node::Variable)
    # Variables are inputs, so we do not compute gradients for them
    nothing
end

end # module
