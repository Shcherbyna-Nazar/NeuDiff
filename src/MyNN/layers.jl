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
    Dense{T, typeof(act)}(W, b, act)
end

function (layer::Dense)(x::MyAD.GraphNode)
    z = MyAD.MatMulOperator(layer.W, x) + layer.b
    MyAD.BroadcastedOperator(layer.activation, z)
end

# === Dropout Layer ===
struct Dropout{T}
    rate::T
end

function Dropout(rate::T) where {T}
    Dropout{T}(rate)
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
    weights = pretrained_weights === nothing ?
        randn(T, embedding_dim, vocab_size) * sqrt(T(1) / vocab_size) :
        pretrained_weights
    w_var = MyAD.Variable(weights, zeros(T, size(weights)...))
    Embedding{T}(w_var)
end

function (layer::Embedding)(x::AbstractMatrix{<:Integer})
    word_idxs = vec(collect(x))
    seq_len, batch_size = size(x)
    shape = (size(layer.weight.output, 1), seq_len, batch_size)
    MyAD.EmbeddingOp(layer.weight, word_idxs, shape)
end

# === Conv1D Layer ===
struct Conv1D{T, F}
    W::MyAD.Variable{T}          # (kernel_size, in_channels, out_channels)
    b::MyAD.Variable{T}          # (out_channels, 1)
    activation::F
end

function Conv1D(in_channels::Int, out_channels::Int, kernel_size::Int, act = MyAD.identity_fn)
    T = Float32
    std = act === MyAD.relu ? sqrt(T(2) / (in_channels * kernel_size)) : sqrt(T(1) / (in_channels * kernel_size))
    W = MyAD.Variable(
        randn(T, kernel_size, in_channels, out_channels) * std,
        zeros(T, kernel_size, in_channels, out_channels)
    )
    b = MyAD.Variable(zeros(T, out_channels, 1), zeros(T, out_channels, 1))
    Conv1D{T, typeof(act)}(W, b, act)
end

function (layer::Conv1D)(x::MyAD.GraphNode)
    MyAD.Conv1DOp(
        layer.W,
        layer.b,
        x,
        size(layer.W.output, 1),  # kernel size (K)
        1,                        # stride
        0,                        # padding
        layer.activation
    )
end

# === MaxPool1D Layer ===
struct MaxPool1D
    kernel_size::Int
    stride::Int
end

function (pool::MaxPool1D)(x::MyAD.GraphNode)
    MyAD.MaxPool1DOp(x, pool.kernel_size, pool.stride)
end

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