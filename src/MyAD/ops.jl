import Base: +, *, -, /, ^

Variable(data::Array{T, N}) where {T, N} = Variable{T, N}(data, zeros(T, size(data)))
Variable(data::AbstractArray{T, N}) where {T, N} =
    Variable{T, N}(Array(data), zeros(T, size(data)))
Variable(data::AbstractArray{T, N}, grad::AbstractArray{T, N}) where {T, N} =
    Variable{T, N}(Array(data), Array(grad))

function ScalarOperator(f::F, a::GraphNode, b::GraphNode) where {F}
    T = promote_type(eltype(a.output), eltype(b.output))
    N = max(ndims(a.output), ndims(b.output))
    empty_shape = ntuple(_ -> 0, N)
    ScalarOperator{F, T, N}(f, (a, b), Array{T}(undef, empty_shape...), Array{T}(undef, empty_shape...))
end

function MatMulOperator(A::GraphNode, B::GraphNode)
    T = promote_type(eltype(A.output), eltype(B.output))
    MatMulOperator{T, ndims(A.output), ndims(B.output)}(A, B, Array{T}(undef, 0, 0), Array{T}(undef, 0, 0))
end

function BroadcastedOperator(f::F, x::GraphNode) where {F}
    T = eltype(x.output)
    shape = size(x.output)
    BroadcastedOperator{F, T, length(shape)}(f, x, zeros(T, shape), zeros(T, shape))
end

function flatten_last_two_dims(x::GraphNode)
    T = eltype(x.output)
    orig_shape = size(x.output)
    out_shape = (:, size(x.output, ndims(x.output)))
    FlattenOp{T, length(orig_shape), 2}(x, orig_shape, Array{T}(undef, 0, 0), Array{T}(undef, 0, 0))
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

function MaxPool1DOp(x::GraphNode, kernel_size::Int, stride::Int)
    T = eltype(x.output)
    MaxPool1DOp{T}(x, kernel_size, stride,
        Array{T, 3}(undef, 0, 0, 0), Array{T, 3}(undef, 0, 0, 0),
        Array{Int, 3}(undef, 0, 0, 0), Array{T, 3}(undef, 0, 0, 0))
end

function PermuteDimsOp(x::GraphNode, dims::NTuple{N, Int}) where {N}
    T = eltype(x.output)
    PermuteDimsOp{T, N}(x, dims, Array{T, N}(undef, 0, 0, 0), Array{T, N}(undef, 0, 0, 0))
end

function EmbeddingOp(weight::Variable{T, 2}, indices::Vector{Int}, shape::NTuple{N, Int}) where {T, N}
    EmbeddingOp{T, N}(weight, indices, shape, Array{T, N}(undef, shape...), Array{T, N}(undef, shape...))
end


Base.:+(a::GraphNode, b::GraphNode) = ScalarOperator(+, a, b)
Base.:*(a::GraphNode, b::GraphNode) = ScalarOperator(*, a, b)
Base.:-(a::GraphNode, b::GraphNode) = ScalarOperator(-, a, b)
Base.:/(a::GraphNode, b::GraphNode) = ScalarOperator(/, a, b)
Base.:^(a::GraphNode, b::GraphNode) = ScalarOperator(^, a, b)
Base.:^(a::GraphNode, b::Number) = ScalarOperator(^, a, Constant(b))