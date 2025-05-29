module MyNN

using ..MyAD

export Dense, Chain, parameters, update!, Dropout, zero_gradients!, AdamState,
       update_adam!, Embedding, Conv1D, MaxPool1D

# Optimized Dense Layer
struct Dense
    W::MyAD.Variable
    b::MyAD.Variable
    activation::Function
end

function Dense(in::Int, out::Int, act = MyAD.identity_fn)
    std = act == MyAD.relu ? sqrt(2 / in) : sqrt(1.0) 

    W = MyAD.Variable(randn(out, in) * std, zeros(out, in))        
    b = MyAD.Variable(zeros(out, 1), zeros(out, 1))
    return Dense(W, b, act)
end

function (layer::Dense)(x::MyAD.GraphNode)
    z = MyAD.MatMulOperator(layer.W, x)
    z = MyAD.ScalarOperator(broadcast_add, z, layer.b)
    return MyAD.BroadcastedOperator(layer.activation, z)
end

# Dropout Layer (No change)
struct Dropout
    rate::Float64
end

function (d::Dropout)(x::MyAD.GraphNode)
    return x  # This can be optimized based on how dropout is used
end

# Chain Model (No change)
struct Chain
    layers::Vector{Any}
end

Chain(args...) = Chain(collect(args))

function (chain::Chain)(x)
    for layer in chain.layers
        x = layer(x)
    end
    return x
end

# Parameters Function
function parameters(model::Chain)
    ps = MyAD.GraphNode[]
    for layer in model.layers
        if layer isa Dense
            push!(ps, layer.W, layer.b)
        end
        if layer isa Conv1D
            push!(ps, layer.W, layer.b)
        end
    end
    return ps
end

# Update Function (In-place memory operation)
function update!(params::Vector{MyAD.GraphNode}, η::Real)
    for p in params
        p.output .-= η .* p.gradient
    end
end

# Zero Gradients Function
function zero_gradients!(model::Chain)
    for p in parameters(model)
        p.gradient .= 0.0
    end
end

# Adam Optimizer State (Optimized)
mutable struct AdamState
    m::Vector{AbstractArray{Float64}}  
    v::Vector{AbstractArray{Float64}} 
    β1::Float64
    β2::Float64
    ϵ::Float64
    t::Int
end

function AdamState(params; β1=0.9, β2=0.999, ϵ=1e-8)
    m = [similar(p.output) .= 0.0 for p in params]
    v = [similar(p.output) .= 0.0 for p in params]
    return AdamState(m, v, β1, β2, ϵ, 0)
end

# Adam Optimizer Update (Optimized)
function update_adam!(state::AdamState, params::Vector{MyAD.GraphNode}, η::Real)
    state.t += 1
    for (i, p) in enumerate(params)
        g = p.gradient
        state.m[i] .= state.β1 .* state.m[i] .+ (1 .- state.β1) .* g
        state.v[i] .= state.β2 .* state.v[i] .+ (1 .- state.β2) .* (g .^ 2)

        m_hat = state.m[i] ./ (1 .- state.β1 ^ state.t)
        v_hat = state.v[i] ./ (1 .- state.β2 ^ state.t)

        p.output .-= η .* m_hat ./ (sqrt.(v_hat) .+ state.ϵ)
    end
end

# Optimized Embedding Layer (using in-place operations)
struct Embedding
    weight::MyAD.Variable  # (embedding_dim, vocab_size)
end

function Embedding(vocab_size::Int, embedding_dim::Int; pretrained_weights=nothing)
    if pretrained_weights === nothing
        weights = randn(embedding_dim, vocab_size) * sqrt(1 / vocab_size)
    else
        weights = pretrained_weights
    end
    w_var = MyAD.Variable(weights, zeros(size(weights)))
    return Embedding(w_var)
end

function (layer::Embedding)(x::Matrix{Int})
    # Flatten and index directly
    word_idxs = vec(x)  # shape: (seq_len * batch_size)
    emb = layer.weight.output[:, word_idxs]  # (embedding_dim, seq_len * batch_size)
    
    # Reshape to (embedding_dim, seq_len, batch_size)
    seq_len, batch_size = size(x)
    output = reshape(emb, size(emb, 1), seq_len, batch_size)
    
    return MyAD.Constant(output)
end

# Optimized Conv1D Layer (using efficient convolutions)
export Conv1D
struct Conv1D
    W::MyAD.Variable  # (out_channels, in_channels, kernel_size)
    b::MyAD.Variable  # (out_channels, 1)
    activation::Function
end

function Conv1D(in_channels::Int, out_channels::Int, kernel_size::Int, act = MyAD.identity_fn)
    std = act == MyAD.relu ? sqrt(2 / (in_channels * kernel_size)) : 1.0
    W = randn(out_channels, in_channels, kernel_size) * std
    b = zeros(out_channels, 1)

    return Conv1D(
        MyAD.Variable(W, zeros(size(W))),
        MyAD.Variable(b, zeros(size(b))),
        act
    )
end

function (layer::Conv1D)(x::MyAD.GraphNode)
    conv = MyAD.Conv1DOp(layer.W, layer.b, x)
    act = MyAD.BroadcastedOperator(layer.activation, conv)
    return act
end

# MaxPool1D Layer
export MaxPool1D
function MaxPool1D(kernel_size::Int, stride::Int)
    return x -> MaxPool1DOp(x, kernel_size, stride, nothing, nothing, nothing)
end

end # module
