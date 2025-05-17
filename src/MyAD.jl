module MyAD

export Variable, Constant, ScalarOperator, topological_sort, forward!, backward!

abstract type GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Float64
    gradient :: Float64
end

mutable struct ScalarOperator{F} <: GraphNode
    f::F
    inputs::Vector{GraphNode}
    output::Float64
    gradient::Float64
end

# Create ScalarOperator from variable number of inputs
ScalarOperator(f, args::GraphNode...) = ScalarOperator{typeof(f)}(f, collect(args), 0.0, 0.0)

# Overload basic operators
import Base: +, *, -, /, sin
+(a::GraphNode, b::GraphNode) = ScalarOperator(+, a, b)
*(a::GraphNode, b::GraphNode) = ScalarOperator(*, a, b)
-(a::GraphNode, b::GraphNode) = ScalarOperator(-, a, b)
/(a::GraphNode, b::GraphNode) = ScalarOperator(/, a, b)
sin(a::GraphNode) = ScalarOperator(sin, a)

# Topological sorting
function visit(node::GraphNode, visited::Set, order::Vector)
    if node ∉ visited
        push!(visited, node)
        if node isa ScalarOperator
            for input in node.inputs
                visit(input, visited, order)
            end
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

# Forward pass
function forward(node::ScalarOperator)
    inputs = map(n -> n.output, node.inputs)
    node.output = node.f(inputs...)
end

function forward!(nodes::Vector{GraphNode})
    for node in nodes
        if node isa ScalarOperator
            forward(node)
        end
    end
end

# Backward pass
function backward(node::ScalarOperator)
    f, inputs = node.f, node.inputs
    out_grad = node.gradient

    if f == +
        for input in inputs
            input.gradient += out_grad
        end
    elseif f == *
        a, b = inputs
        a.gradient += out_grad * b.output
        b.gradient += out_grad * a.output
    elseif f == sin
        x = inputs[1]
        x.gradient += out_grad * cos(x.output)
    end
end

function backward!(nodes::Vector{GraphNode}, seed=1.0)
    last(nodes).gradient = seed
    for node in reverse(nodes)
        if node isa ScalarOperator
            backward(node)
        end
    end
end

end # module
