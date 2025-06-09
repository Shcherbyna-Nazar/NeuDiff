module MyNN

using ..MyAD

export Dense, Chain, parameters, update!, Dropout, zero_gradients!, AdamState,
    update_adam!, Embedding, Conv1D, MaxPool1D

# === Dense Layer ===
struct Dense{T, F}
    W::MyAD.Variable{T}
    b::MyAD.Variable{T}
    activation::F
end

function Dense(in::Int, out::Int, act = MyAD.identity_fn)
    T = Float32
    std = act === MyAD.relu ? sqrt(T(2) / in) : sqrt(T(1) / in)
    W = MyAD.Variable(randn(T, out, in) * std, zeros(T, out, in))
    b = MyAD.Variable(zeros(T, out, 1), zeros(T, out, 1))
    Dense(W, b, act)
end

function (layer::Dense)(x::MyAD.GraphNode)
    z = MyAD.MatMulOperator(layer.W, x) + layer.b
    MyAD.BroadcastedOperator(layer.activation, z)
end

# === Dropout Layer ===
struct Dropout{T}
    rate::T
end

function (d::Dropout)(x::MyAD.GraphNode)
    # Placeholder; implement dropout logic if needed
    x
end

# === Embedding Layer ===
struct Embedding{T}
    weight::MyAD.Variable{T}  # (embedding_dim, vocab_size)
end

function Embedding(vocab_size::Int, embedding_dim::Int; pretrained_weights=nothing)
    T = Float32
    weights = pretrained_weights === nothing ? randn(T, embedding_dim, vocab_size) * sqrt(T(1) / vocab_size) : pretrained_weights
    w_var = MyAD.Variable(weights, zeros(T, size(weights)))
    Embedding(w_var)
end

function (layer::Embedding)(x::AbstractMatrix{<:Integer})
    word_idxs = vec(collect(x))
    seq_len, batch_size = size(x)
    shape = (size(layer.weight.output, 1), seq_len, batch_size)
    MyAD.EmbeddingOp(layer.weight, word_idxs, shape)
end

# === Conv1D Layer ===
struct Conv1D{T, F}
    W::MyAD.Variable{T}  # (out_channels, in_channels, kernel_size)
    b::MyAD.Variable{T}  # (out_channels, 1)
    activation::F
end

function Conv1D(in_channels::Int, out_channels::Int, kernel_size::Int, act = MyAD.identity_fn)
    T = Float32
    std = act === MyAD.relu ? sqrt(T(2) / (in_channels * kernel_size)) : sqrt(T(1) / (in_channels * kernel_size))
    W = MyAD.Variable(randn(T, out_channels, in_channels, kernel_size) * std, zeros(T, out_channels, in_channels, kernel_size))
    b = MyAD.Variable(zeros(T, out_channels, 1), zeros(T, out_channels, 1))
    Conv1D(W, b, act)
end

function (layer::Conv1D)(x::MyAD.GraphNode)
    MyAD.Conv1DOp(layer.W, layer.b, x,
                  size(layer.W.output, 3),  # kernel size
                  1,                        # stride
                  0,                        # padding
                  layer.activation)
end

# === MaxPool1D Layer ===
MaxPool1D(kernel_size::Int, stride::Int) = x -> MyAD.MaxPool1DOp(x, kernel_size, stride)

# === Chain Model ===
struct Chain{L}
    layers::L
end

Chain(args...) = Chain(args)

function (chain::Chain)(x)
    for layer in chain.layers
        x = layer(x)
    end
    x
end

# === Parameter Utilities ===
function parameters(model::Chain)
    ps = MyAD.GraphNode[]
    for layer in model.layers
        if layer isa Dense || layer isa Conv1D
            push!(ps, layer.W, layer.b)
        elseif layer isa Embedding
            push!(ps, layer.weight)
        end
    end
    ps
end

function zero_gradients!(model::Chain)
    for p in parameters(model)
        fill!(p.gradient, 0)
    end
end

# === SGD Update ===
function update!(params::Vector{<:MyAD.GraphNode}, η::Real)
    for p in params
        @. p.output -= η * p.gradient
    end
end

# === Adam Optimizer ===
mutable struct AdamState{T}
    m::Vector{Array{T}}
    v::Vector{Array{T}}
    β1::T
    β2::T
    ϵ::T
    t::Int
end

function AdamState(params; β1=0.9, β2=0.999, ϵ=1e-8)
    T = eltype(params[1].output)
    m = [zeros(T, size(p.output)) for p in params]
    v = [zeros(T, size(p.output)) for p in params]
    AdamState{T}(m, v, β1, β2, ϵ, 0)
end

function update_adam!(state::AdamState, params::Vector{<:MyAD.GraphNode}, η::Real)
    state.t += 1
    for (i, p) in enumerate(params)
        g = p.gradient
        m, v = state.m[i], state.v[i]

        @. m = state.β1 * m + (1 - state.β1) * g
        @. v = state.β2 * v + (1 - state.β2) * g^2

        m_hat = m ./ (1 - state.β1 ^ state.t)
        v_hat = v ./ (1 - state.β2 ^ state.t)

        @. p.output -= η * m_hat / (sqrt(v_hat) + state.ϵ)
    end
end

end # module
