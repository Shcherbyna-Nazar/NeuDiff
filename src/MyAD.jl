module MyAD

export Variable, Constant, ScalarOperator, MatMulOperator, BroadcastedOperator,
       topological_sort, forward!, backward!, Dense, relu, sigmoid

# === Abstract base ===
abstract type GraphNode end

# === Basic nodes ===
struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
end

# === Operators ===
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

# === Activations ===
relu(x) = max.(0, x)
sigmoid(x) = 1.0 ./ (1.0 .+ exp.(-x))
identity_fn(x) = x

# === Constructors ===
ScalarOperator(f, args::GraphNode...) = ScalarOperator{typeof(f)}(f, collect(args), nothing, nothing)
BroadcastedOperator(f::F, x::GraphNode) where {F} = BroadcastedOperator{F}(f, x, nothing, nothing)

# === Operator overloads ===
import Base: +, *, -, /, sin
+(a::GraphNode, b::GraphNode) = ScalarOperator(+, a, b)
*(a::GraphNode, b::GraphNode) = ScalarOperator(*, a, b)
-(a::GraphNode, b::GraphNode) = ScalarOperator(-, a, b)
/(a::GraphNode, b::GraphNode) = ScalarOperator(/, a, b)
sin(x::GraphNode) = ScalarOperator(sin, x)

# === Topological sort ===
function visit(node::GraphNode, visited::Set, order::Vector)
    if node ∉ visited
        push!(visited, node)
        if node isa ScalarOperator
            for input in node.inputs
                visit(input, visited, order)
            end
        elseif node isa MatMulOperator
            visit(node.A, visited, order)
            visit(node.B, visited, order)
        elseif node isa BroadcastedOperator
            visit(node.input, visited, order)
        end
        push!(order, node)
    end
end

function topological_sort(root::GraphNode)
    visited = Set()
    order = Vector{GraphNode}()
    visit(root, visited, order)
    return order
end

# === Forward pass ===
function forward(node::ScalarOperator)
    inputs = map(n -> n.output, node.inputs)
    node.output = node.f(inputs...)
    # println("Forward ScalarOperator: inputs=$(map(size, inputs)), output=$(size(node.output))")
end

function forward(node::MatMulOperator)
    node.output = node.A.output * node.B.output
    # println("Forward MatMul: A=$(size(node.A.output)), B=$(size(node.B.output)), output=$(size(node.output))")
end

function forward(node::BroadcastedOperator)
    node.output = node.f.(node.input.output)
    # println("Forward Broadcasted: input=$(size(node.input.output)), output=$(size(node.output)), func=$(node.f)")
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

# === Backward pass ===
function backward(node::ScalarOperator)
    f, inputs = node.f, node.inputs
    out_grad = node.gradient
    # println("Backward ScalarOperator: f=$f, grad=$(size(out_grad))")
    if f === +
        for input in inputs
            input.gradient .+= out_grad
        end
    elseif f === broadcast_add
        a, b = inputs
        a.gradient .+= out_grad
        b.gradient .+= sum(out_grad, dims=2)
    elseif f === *
        a, b = inputs
        a.gradient .+= out_grad .* b.output
        b.gradient .+= out_grad .* a.output
    elseif f === -
        a, b = inputs
        a.gradient .+= out_grad
        b.gradient .-= out_grad
    elseif f === /
        a, b = inputs
        a.gradient .+= out_grad ./ b.output
        b.gradient .-= out_grad .* a.output ./ (b.output .^ 2)
    elseif f === sin
        x = inputs[1]
        x.gradient .+= out_grad .* cos.(x.output)
    else
        println("  Unknown function $f in backward")
    end
end


function backward(node::MatMulOperator)
    A, B = node.A, node.B
    out_grad = node.gradient
    # println("Backward MatMul: out_grad=$(size(out_grad))")
    A.gradient .+= out_grad * B.output'
    B.gradient .+= A.output' * out_grad
end

function backward(node::BroadcastedOperator)
    f, x = node.f, node.input
    out_grad = node.gradient
    # println("Backward Broadcasted: f=$f, grad=$(size(out_grad)), input=$(size(x.output))")
    if f === relu
        x.gradient .+= out_grad .* (x.output .> 0)
    elseif f === identity_fn
        x.gradient .+= out_grad
    elseif f === sigmoid
        σ = sigmoid(x.output)
        x.gradient .+= out_grad .* σ .* (1 .- σ)
    elseif f === tanh
        x.gradient .+= out_grad .* (1 .- tanh.(x.output).^2)
    else
        println("  Unknown broadcasted function $f in backward")
    end
end


function backward!(nodes::Vector{GraphNode}, seed=1.0)
    # println("=== Starting backward! ===")
    for node in nodes
        if !isnothing(node.output)
            node.gradient = zeros(size(node.output))
        end
    end

    last(nodes).gradient = seed
    # println("Seed gradient set to size: ", size(seed isa Number ? [seed] : seed))

    for node in reverse(nodes)
        # println("Backpropagating through node: ", typeof(node))
        if node isa ScalarOperator
            backward(node)
        elseif node isa MatMulOperator
            backward(node)
        elseif node isa BroadcastedOperator
            backward(node)
        end
    end
end


# === Dense layer ===
# === Dense layer with Xavier initialization ===
struct Dense
    W::Variable
    b::Variable
    activation::Function
end

function Dense(in::Int, out::Int, act=identity_fn)
    limit = sqrt(6.0 / (in + out))
    W = Variable(rand(Float64, out, in) .* (2limit) .- limit, zeros(out, in))
    b = Variable(zeros(out, 1), zeros(out, 1))
    return Dense(W, b, act)
end

function broadcast_add(a,b)
    return a .+ b
end

function (layer::Dense)(x::GraphNode)
    z = MatMulOperator(layer.W, x, nothing, nothing)
    z = ScalarOperator(broadcast_add, z, layer.b)
    return BroadcastedOperator(layer.activation, z)
end


end # module
