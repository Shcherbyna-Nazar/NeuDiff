module MyAD

export GraphNode, Constant, Variable, ScalarOperator, MatMulOperator, BroadcastedOperator,
    forward!, backward!, topological_sort, relu, sigmoid, identity_fn, broadcast_add

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
    output::Any
    gradient::Any
end

function Conv1DOp(W::Variable, b::Variable, x::GraphNode)
    _, _, k = size(W.output)
    Conv1DOp(W, b, x, k, nothing, nothing)
end


using FFTW

FFTW.set_num_threads(Sys.CPU_THREADS)

using Base.Threads

function forward(node::Conv1DOp)
    W, b, x = node.W.output, node.b.output, node.x.output
    seq_len, in_ch, batch = size(x)
    out_ch, _, k = size(W)

    fft_len = nextpow(2, seq_len + k - 1)
    out_len = seq_len - k + 1

    # Allocate output once
    node.output === nothing && (node.output = zeros(out_ch, out_len, batch))
    out = node.output

    @inbounds Threads.@threads for b_id in 1:batch
        # Thread-local buffers
        x_pad = zeros(fft_len)
        w_pad = zeros(fft_len)
        sum_fft = zeros(fft_len)
        plan_x = plan_rfft(x_pad)
        plan_w = plan_rfft(w_pad)
        dummy_fft = plan_x * x_pad
        plan_ir = plan_irfft(dummy_fft, fft_len)

        for o in 1:out_ch
            sum_fft .= 0.0
            for i in 1:in_ch
                # Prepare padded input and kernel
                x_pad[1:seq_len] .= x[:, i, b_id]
                x_pad[(seq_len+1):end] .= 0.0

                w_pad[1:k] .= reverse(W[o, i, :])
                w_pad[(k+1):end] .= 0.0

                x_fft = plan_x * x_pad
                w_fft = plan_w * w_pad
                conv_fft = x_fft .* w_fft

                conv_time = plan_ir * conv_fft
                sum_fft .+= conv_time
            end
            out[o, :, b_id] .= sum_fft[k : k + out_len - 1] .+ b[o]
        end
    end
end


function backward(node::Conv1DOp)
    W, b, x = node.W, node.b, node.x
    dy = node.gradient
    x_val, W_val = x.output, W.output

    seq_len, in_ch, batch = size(x_val)
    out_ch, _, k = size(W_val)
    out_len = size(dy, 2)
    fft_len = nextpow(2, seq_len + k - 1)

    dx_total = zeros(in_ch, seq_len, batch)
    dW_partials = Vector{Array{Float64, 3}}(undef, batch)
    db_partials = Vector{Array{Float64, 1}}(undef, batch)

    Threads.@threads for b_id in 1:batch
        # Thread-local buffers
        x_pad = zeros(fft_len)
        w_pad = zeros(fft_len)
        dy_pad = zeros(fft_len)

        dx_local = zeros(in_ch, seq_len)
        dW_local = zeros(out_ch, in_ch, k)
        db_local = zeros(out_ch)

        plan_x = plan_rfft(x_pad)
        plan_w = plan_rfft(w_pad)
        plan_ir = plan_irfft(plan_x * x_pad, fft_len)

        for o in 1:out_ch
            dy_pad[1:out_len] .= dy[o, :, b_id]
            dy_pad[(out_len+1):end] .= 0.0
            dy_fft = plan_x * dy_pad

            for i in 1:in_ch
                # dW[o, i, :] += reverse(conv(x[i], dy[o]))
                x_pad[1:seq_len] .= x_val[i, :, b_id]
                x_pad[(seq_len+1):end] .= 0.0
                x_fft = plan_x * x_pad

                dW_fft = dy_fft .* x_fft
                dW_time = plan_ir * dW_fft
                dW_local[o, i, :] .+= reverse(dW_time[1:k])

                # dx[i] += conv(dy[o], W[o][i])
                w_pad[1:k] .= W_val[o, i, :]
                w_pad[(k+1):end] .= 0.0
                w_fft = plan_w * w_pad

                dx_fft = dy_fft .* w_fft
                dx_time = plan_ir * dx_fft
                dx_local[i, :] .+= dx_time[k:k + seq_len - 1]
            end

            db_local[o] += sum(dy[o, :, b_id])
        end

        dx_total[:, :, b_id] .= dx_local
        dW_partials[b_id] = dW_local
        db_partials[b_id] = db_local
    end

    # Serial reduction after threads
    dW_total = zeros(out_ch, in_ch, k)
    db_total = zeros(out_ch, 1)
    for b_id in 1:batch
        dW_total .+= dW_partials[b_id]
        db_total[:, 1] .+= db_partials[b_id]
    end

    W.gradient = dW_total
    b.gradient = db_total
    x.gradient = dx_total
end



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
