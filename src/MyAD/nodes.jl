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


# === Operator Nodes ===
mutable struct ScalarOperator{F, T, N} <: GraphNode
    f::F
    inputs::NTuple{2, GraphNode}
    output::Array{T, N}
    gradient::Array{T, N}
end


mutable struct MatMulOperator{T, NA, NB} <: GraphNode
    A::GraphNode
    B::GraphNode
    output::Array{T, 2}
    gradient::Array{T, 2}
end


mutable struct BroadcastedOperator{F, T, N} <: GraphNode
    f::F
    input::GraphNode
    output::Array{T, N}
    gradient::Array{T, N}
end


mutable struct FlattenOp{T, N, NO} <: GraphNode
    x::GraphNode
    orig_shape::NTuple{N, Int}
    output::Array{T, NO}
    gradient::Array{T, NO}
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


mutable struct MaxPool1DOp{T} <: GraphNode
    x::GraphNode
    kernel_size::Int
    stride::Int
    output::Array{T, 3}
    gradient::Array{T, 3}
    indices::Array{Int, 3}
    dx::Array{T, 3}
end


mutable struct PermuteDimsOp{T, N} <: GraphNode
    x::GraphNode
    dims::NTuple{N, Int}
    output::Array{T, N}
    gradient::Array{T, N}
end


mutable struct EmbeddingOp{T, N} <: GraphNode
    weight::Variable{T, 2}
    indices::Vector{Int}
    shape::NTuple{N, Int}
    output::Array{T, N}
    gradient::Array{T, N}
end