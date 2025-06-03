include("src/MyAD.jl")
include("src/MyNN.jl")
using .MyAD, .MyNN
using JLD2, Printf, Statistics, Random
using TimerOutputs
using Profile
using ProfileView  # GUI flame chart visualization
using LinearAlgebra

const TO = TimerOutput()

# === Load IMDB and GloVe data ===
X_train = load("data/imdb_dataset_prepared.jld2", "X_train")
y_train = load("data/imdb_dataset_prepared.jld2", "y_train")
embeddings = load("data/imdb_dataset_prepared.jld2", "embeddings")
vocab = load("data/imdb_dataset_prepared.jld2", "vocab")

# Ensure X_train and X_test are of type Matrix{Int}
X_train = Matrix{Int}(X_train)  # Convert to Matrix{Int}
X_test = Matrix{Int}(load("data/imdb_dataset_prepared.jld2", "X_test"))

y_test = load("data/imdb_dataset_prepared.jld2", "y_test")

embedding_dim = size(embeddings, 1)
vocab_size = length(vocab)

embedding = Embedding(vocab_size, embedding_dim; pretrained_weights=embeddings)
permute = x -> PermuteDimsOp(x, (2, 1, 3))
conv = Conv1D(embedding_dim, 8, 3, relu)
pool = MaxPool1D(8, 8)
flatten = flatten_last_two_dims
dense = Dense(128, 1, sigmoid)

model = Chain(embedding, permute, conv, pool, flatten, dense)


# === Define loss and accuracy ===
function bce(ŷ, y)
    ϵ = 1e-7
    ŷ_clipped = clamp.(ŷ, ϵ, 1 .- ϵ)
    return -mean(y .* log.(ŷ_clipped) .+ (1 .- y) .* log.(1 .- ŷ_clipped))
end

function bce_grad(ŷ, y)
    return (ŷ .- y) ./ length(y)
end




accuracy(ŷ, y) = mean((ŷ .> 0.5) .== (y .> 0.5))

# === Training settings ===
params = parameters(model)
# Try fine-tuning Adam 
state = AdamState(params)
epochs = 5
η = 0.001
batch_size = 64


function create_batches(X, Y; batchsize=64, shuffle=true)
    idxs = collect(1:size(X, 2))
    if shuffle
        Random.shuffle!(idxs)
    end
    return [(X[:, idxs[i:min(i+batchsize-1, end)]],
             Y[:, idxs[i:min(i+batchsize-1, end)]] )
             for i in 1:batchsize:length(idxs)]
end

# === Training loop ===
for epoch in 1:epochs
    println("=== Epoch $epoch ===")
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    batches = create_batches(X_train, y_train, batchsize=batch_size)
    println("  → Training on $(length(batches)) batches of size $batch_size...")

    t = @elapsed begin
        for (i, (x, y)) in enumerate(batches)
            # println("  → Batch $i")

            out = model(x)
            graph = topological_sort(out)

            # println("    Forward pass...")
            forward!(graph)

            ŷ = out.output
            loss = bce(ŷ, y)
            acc = accuracy(ŷ, y)

            total_loss += loss
            total_acc += acc
            num_batches += 1

            # println(@sprintf("    Train loss: %.4f | acc: %.4f", loss, acc))

            # Zakładając, że ostatnia warstwa to sigmoid + BCE, gradient to (ŷ - y)
            zero_gradients!(model)
            out.gradient = bce_grad(ŷ, y)
            backward!(graph, out.gradient)
            for p in parameters(model)
                if any(isnan.(p.gradient))
                    @warn "NaN in gradient for parameter $(p.name)!" p.gradient
                end
            end

            update_adam!(state, params, η)

            if i % 100 == 0
                println(@sprintf("    Batch %d/%d: loss = %.4f, acc = %.4f", i, length(batches), loss, acc))
            end
        end
    end

    train_loss = total_loss / num_batches
    train_acc = total_acc / num_batches

    println("  → Evaluation on test set...")
    
    # Ensure X_test is in the correct type for Embedding layer
    out_eval = model(Matrix{Int}(X_test))  # Ensure X_test is in Matrix{Int} format
    # Ensure the output is a Variable for evaluation
    forward!(topological_sort(out_eval))
    test_pred = out_eval.output
    test_loss = bce(test_pred, y_test)
    test_acc = accuracy(test_pred, y_test)
    println("Example predictions: ", round.(test_pred[1:10]; digits=3))
    println("Ground truth       : ", y_test[1,1:10])


    println(@sprintf("✅ Epoch %d finished in %.2fs", epoch, t))
    println(@sprintf("🏋️  Train: loss = %.4f, acc = %.4f", train_loss, train_acc))
    println(@sprintf("🧪  Test : loss = %.4f, acc = %.4f\n", test_loss, test_acc))
end
