module MyAD

export GraphNode, Constant, Variable, ScalarOperator, MatMulOperator, BroadcastedOperator,
       forward!, backward!, topological_sort, relu, sigmoid, identity_fn, broadcast_add

# === Abstract Node Type ===
abstract type GraphNode end

# === Basic Nodes ===
struct Constant{T} <: GraphNode
    output :: T
    Constant(x::T) where {T} = new{T}(x)
end



mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
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

# === Constructors ===
ScalarOperator(f, args::GraphNode...) = ScalarOperator{typeof(f)}(f, collect(args), nothing, nothing)
BroadcastedOperator(f::F, x::GraphNode) where {F} = BroadcastedOperator{F}(f, x, nothing, nothing)

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

function forward!(nodes::Vector{GraphNode})
    for node in nodes
        if node isa ScalarOperator
            forward(node)
        elseif node isa MatMulOperator
            forward(node)
        elseif node isa BroadcastedOperator
            forward(node)
        end
    end
end

function broadcast_add(a::AbstractMatrix, b::AbstractMatrix)
    return a .+ b
end

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
        x.gradient .+= out_grad .* (x.output .> 0)
    elseif f == identity_fn
        x.gradient .+= out_grad
    elseif f == sigmoid
        σ = sigmoid(x.output)
        x.gradient .+= out_grad .* σ .* (1 .- σ)
    elseif f == tanh
        x.gradient .+= out_grad .* (1 .- tanh.(x.output).^2)
    else
        error("Unsupported function in BroadcastedOperator backward: $f")
    end
end

function backward!(nodes::Vector{GraphNode}, seed=1.0)
    for node in nodes
        if !isnothing(node.output)
            node.gradient = zeros(size(node.output))
        end
    end
    last(nodes).gradient = seed
    for node in reverse(nodes)
        if node isa ScalarOperator
            backward(node)
        elseif node isa MatMulOperator
            backward(node)
        elseif node isa BroadcastedOperator
            backward(node)
        end
    end
end


end # module
