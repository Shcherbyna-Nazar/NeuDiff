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

# === Conv1D Operation ===
mutable struct Conv1DOp <: GraphNode
    W::Variable                     # (out_ch, in_ch, kernel)
    b::Variable                     # (out_ch, 1)
    x::GraphNode                    # (in_ch, seq_len, batch)
    kernel_size::Int
    output::Any
    gradient::Any
end

function Conv1DOp(W::Variable, b::Variable, x::GraphNode)
    _, _, k = size(W.output)
    Conv1DOp(W, b, x, k, nothing, nothing)
end

using FFTW
using FFTW:mul!

# === Optimized Conv1D Forward Pass with FFT ===
function forward(node::Conv1DOp)
    W, b, x = node.W.output, node.b.output, node.x.output
    in_ch, seq_len, batch = size(x)
    out_ch, _, k = size(W)

    fft_len = nextpow(2, seq_len + k - 1)  # Optimized FFT length
    out_len = seq_len - k + 1

    node.output === nothing && (node.output = zeros(out_ch, out_len, batch))
    out = node.output

    Threads.@threads for b_id in 1:batch
        x_pad = zeros(fft_len)
        w_pad = zeros(fft_len)
        sum_fft = zeros(fft_len)

        x_fft = similar(plan_rfft(x_pad) * x_pad)
        w_fft = similar(x_fft)
        conv_fft = similar(x_fft)
        conv_time = zeros(fft_len)

        plan_x = plan_rfft(x_pad)
        plan_w = plan_rfft(w_pad)
        plan_ir = plan_irfft(conv_fft, fft_len)

        @views x_b = x[:, :, b_id]
        @views out_b = out[:, :, b_id]

        for o in 1:out_ch
            sum_fft .= 0.0
            for i in 1:in_ch
                @views x_pad[1:seq_len] .= x_b[i, :]
                fill!(x_pad[seq_len+1:end], 0.0)
                @views w_pad[1:k] .= reverse(W[o, i, :])
                fill!(w_pad[k+1:end], 0.0)

                mul!(x_fft, plan_x, x_pad)
                mul!(w_fft, plan_w, w_pad)
                @. conv_fft = x_fft * w_fft
                mul!(conv_time, plan_ir, conv_fft)

                @views sum_fft .+= conv_time
            end
            @views out_b[o, :] .= sum_fft[k : k + out_len - 1] .+ b[o]
        end
    end
end

# === Optimized Conv1D Backward Pass ===
function backward(node::Conv1DOp)
    W, b, x = node.W, node.b, node.x
    dy = node.gradient
    x_val, W_val = x.output, W.output

    in_ch, seq_len, batch = size(x_val)
    out_ch, _, k = size(W_val)
    out_len = size(dy, 2)
    fft_len = nextpow(2, seq_len + k - 1)

    dx = zeros(in_ch, seq_len, batch)
    dW = zeros(out_ch, in_ch, k)
    db = zeros(out_ch, 1)

    dW_partials = Vector{Array{Float64, 3}}(undef, batch)
    db_partials = Vector{Vector{Float64}}(undef, batch)

    Threads.@threads for b_id in 1:batch
        x_pad = zeros(fft_len)
        w_pad = zeros(fft_len)
        dy_pad = zeros(fft_len)

        x_fft = similar(plan_rfft(x_pad) * x_pad)
        w_fft = similar(x_fft)
        dy_fft = similar(x_fft)
        dx_fft = similar(x_fft)
        dW_fft = similar(x_fft)
        dx_time = zeros(fft_len)
        dW_time = zeros(fft_len)

        plan_x = plan_rfft(x_pad)
        plan_w = plan_rfft(w_pad)
        plan_ir = plan_irfft(dW_fft, fft_len)

        @views x_b = x_val[:, :, b_id]
        @views dy_b = dy[:, :, b_id]

        local_dx = zeros(in_ch, seq_len)
        local_dW = zeros(out_ch, in_ch, k)
        local_db = zeros(out_ch)

        for o in 1:out_ch
            @views dy_pad[1:out_len] .= dy_b[o, :]
            fill!(dy_pad[out_len+1:end], 0.0)
            mul!(dy_fft, plan_x, dy_pad)

            for i in 1:in_ch
                @views x_pad[1:seq_len] .= x_b[i, :]
                fill!(x_pad[seq_len+1:end], 0.0)
                mul!(x_fft, plan_x, x_pad)

                @. dW_fft = dy_fft * x_fft
                mul!(dW_time, plan_ir, dW_fft)
                @views local_dW[o, i, :] .+= reverse(dW_time[1:k])

                @views w_pad[1:k] .= W_val[o, i, :]
                fill!(w_pad[k+1:end], 0.0)
                mul!(w_fft, plan_w, w_pad)

                @. dx_fft = dy_fft * w_fft
                mul!(dx_time, plan_ir, dx_fft)
                @views local_dx[i, :] .+= dx_time[k : k + seq_len - 1]
            end

            local_db[o] += sum(dy_b[o, :])
        end

        dx[:, :, b_id] .= local_dx
        dW_partials[b_id] = local_dW
        db_partials[b_id] = local_db
    end

    # Redukcja sumaryczna po wszystkich wątkach
    for b_id in 1:batch
        dW .+= dW_partials[b_id]
        db[:, 1] .+= db_partials[b_id]
    end

    W.gradient = dW
    b.gradient = db
    x.gradient = dx
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
