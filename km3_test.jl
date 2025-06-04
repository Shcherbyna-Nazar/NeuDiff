include("src/MyAD.jl")
include("src/MyNN.jl")
using .MyAD, .MyNN
using JLD2, Printf, Statistics, Random
using TimerOutputs, LinearAlgebra

const TO = TimerOutput()

# === Load data ===
X_train = Int.(load("data/imdb_dataset_prepared.jld2", "X_train"))
y_train = reshape(Float32.(load("data/imdb_dataset_prepared.jld2", "y_train")), 1, :)
X_test = Int.(load("data/imdb_dataset_prepared.jld2", "X_test"))
y_test = reshape(Float32.(load("data/imdb_dataset_prepared.jld2", "y_test")), 1, :)

embeddings = load("data/imdb_dataset_prepared.jld2", "embeddings")
vocab = load("data/imdb_dataset_prepared.jld2", "vocab")

embedding_dim = size(embeddings, 1)
vocab_size = length(vocab)

# === Model ===
model = Chain(
    Embedding(vocab_size, embedding_dim),
    x -> PermuteDimsOp(x, (2, 1, 3)),  # (L, C, B) -> (C, L, B)
    Conv1D(embedding_dim, 8, 3, relu),
    MaxPool1D(8, 8),
    flatten_last_two_dims,
    Dense(128, 1, sigmoid)
)
model.layers[1].weight.output = embeddings  # Set pretrained weights

# === Loss and accuracy ===
function bce(ŷ, y)
    ϵ = 1e-7
    ŷ_clipped = clamp.(ŷ, ϵ, 1 .- ϵ)
    return -mean(y .* log.(ŷ_clipped) .+ (1 .- y) .* log.(1 .- ŷ_clipped))
end

function bce_grad(ŷ, y)
    ϵ = 1e-7
    ŷ_clipped = clamp.(ŷ, ϵ, 1 .- ϵ)
    return (ŷ_clipped .- y) ./ (ŷ_clipped .* (1 .- ŷ_clipped) * size(ŷ, 2))
end

accuracy(ŷ, y) = mean((ŷ .> 0.5) .== (y .> 0.5))

# === Optimizer ===
params = parameters(model)
state = AdamState(params)
η = 0.001
epochs = 5
batch_size = 64

# === Mini-batch generator ===
function create_batches(X, Y; batchsize=64, shuffle=true)
    idxs = collect(1:size(X, 2))
    if shuffle
        Random.shuffle!(idxs)
    end
    return [(X[:, idxs[i:min(i+batchsize-1, end)]],
             Y[:, idxs[i:min(i+batchsize-1, end)]])
             for i in 1:batchsize:length(idxs)]
end

# === Training loop ===
for epoch in 1:epochs
    println("=== Epoch $epoch ===")
    total_loss, total_acc, num_batches = 0.0, 0.0, 0
    batches = create_batches(X_train, y_train, batchsize=batch_size)
    println("  → Training on $(length(batches)) batches of size $batch_size...")

    t = @elapsed begin
        for (i, (x, y)) in enumerate(batches)
            out = model(x)
            graph = topological_sort(out)
            forward!(graph)

            ŷ = out.output
            loss = bce(ŷ, y)
            acc = accuracy(ŷ, y)

            total_loss += loss
            total_acc += acc
            num_batches += 1

            zero_gradients!(model)
            out.gradient = bce_grad(ŷ, y)
            backward!(graph, out.gradient)
            update_adam!(state, params, η)

            if i % 100 == 0 || i == length(batches)
                println(@sprintf("    Batch %d/%d: loss = %.4f, acc = %.4f", i, length(batches), loss, acc))
            end
        end
    end

    train_loss = total_loss / num_batches
    train_acc = total_acc / num_batches

    # === Evaluation ===
    println("  → Evaluation on test set...")
    out_eval = model(X_test)
    forward!(topological_sort(out_eval))
    test_pred = out_eval.output
    test_loss = bce(test_pred, y_test)
    test_acc = accuracy(test_pred, y_test)

    println("🟢 Example predictions: ", round.(test_pred[1:10]; digits=3))
    println("🎯 Ground truth       : ", y_test[1,1:10])
    println(@sprintf("✅ Epoch %d finished in %.2fs", epoch, t))
    println(@sprintf("🏋️  Train: loss = %.4f, acc = %.4f", train_loss, train_acc))
    println(@sprintf("🧪  Test : loss = %.4f, acc = %.4f\n", test_loss, test_acc))
end
