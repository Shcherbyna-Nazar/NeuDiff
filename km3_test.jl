include("src/MyAD.jl")
include("src/MyNN.jl")
using .MyAD, .MyNN
using JLD2, Printf, Statistics, Random
using TimerOutputs
const TO = TimerOutput()


# === Load IMDB and GloVe data ===
X_train = load("data/imdb_dataset_prepared.jld2", "X_train")
y_train = load("data/imdb_dataset_prepared.jld2", "y_train")
X_test  = load("data/imdb_dataset_prepared.jld2", "X_test")
y_test  = load("data/imdb_dataset_prepared.jld2", "y_test")
embeddings = load("data/imdb_dataset_prepared.jld2", "embeddings")
vocab = load("data/imdb_dataset_prepared.jld2", "vocab")

embedding_dim = size(embeddings, 1)
vocab_size = length(vocab)

# === Define model ===
model = Chain(
    Embedding(vocab_size, embedding_dim; pretrained_weights=embeddings),
    Conv1D(embedding_dim, 8, 3, relu),
    MaxPool1D(8, 8),
    flatten_last_two_dims,
    Dense(128, 1, sigmoid)
)


# === Define loss and accuracy ===
function bce(ŷ, y)
    ϵ = 1e-7
    ŷ_clipped = clamp.(ŷ, ϵ, 1 .- ϵ)
    return -mean(y .* log.(ŷ_clipped) .+ (1 .- y) .* log.(1 .- ŷ_clipped))
end

function bce_grad(ŷ, y)
    ϵ = 1e-7
    return (ŷ .- y) ./ clamp.(ŷ .* (1 .- ŷ), ϵ, 1.0) ./ size(y, 2)
end

accuracy(ŷ, y) = mean((ŷ .> 0.5) .== (y .> 0.5))

# === Training settings ===
params = parameters(model)
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
             Y[:, idxs[i:min(i+batchsize-1, end)]])
             for i in 1:batchsize:length(idxs)]
end

# === Training loop ===
for epoch in 1:epochs
    println("=== Epoch $epoch ===")
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    batches = create_batches(X_train, y_train, batchsize=batch_size)[1:10]
    println("  → Training on $(length(batches)) batches of size $batch_size...")

    t = @elapsed begin
        for (i, (x, y)) in enumerate(batches)
            println("  → Batch $i")
            y_node = Variable(y, zeros(size(y)))
            @timeit TO "forward pass" begin
                out = model(x)
                graph = topological_sort(out)
                forward!(graph)
                for node in graph
                    if node isa MyAD.GraphNode
                        in_shape = try
                            node isa ScalarOperator ? join(map(n -> string(size(n.output)), node.inputs), ", ") :
                            node isa MatMulOperator ? "$(size(node.A.output)), $(size(node.B.output))" :
                            node isa BroadcastedOperator ? "$(size(node.input.output))" :
                            node isa Conv1DOp ? "$(size(node.x.output))" :
                            node isa MaxPool1DOp ? "$(size(node.x.output))" :
                            "?"
                        catch
                            "?"
                        end

                        out_shape = !isnothing(node.output) ? string(size(node.output)) : "none"
                        println("✔️ ", typeof(node), " in: ", in_shape, " → out: ", out_shape)
                    end
                end

            end

            ŷ = out.output
            loss = bce(ŷ, y)
            acc = accuracy(ŷ, y)

            total_loss += loss
            total_acc += acc
            num_batches += 1

            println(@sprintf("    Train loss: %.4f | acc: %.4f", loss, acc))

            out.gradient = bce_grad(ŷ, y)
            zero_gradients!(model)

            @timeit TO "backward pass" backward!(graph, out.gradient)
            @timeit TO "update params" update_adam!(state, params, η)
        end
    end

    train_loss = total_loss / num_batches
    train_acc = total_acc / num_batches

    println("  → Evaluation on test set...")
    @timeit TO "eval forward" begin
        out_eval = model(Matrix(X_test))  # jawnie konwertuj na Matrix, jeśli X_test' zostało ztransponowane

        forward!(topological_sort(out_eval))
    end
    test_pred = out_eval.output
    test_loss = bce(test_pred, y_test)
    test_acc = accuracy(test_pred, y_test)

    println(@sprintf("✅ Epoch %d finished in %.2fs", epoch, t))
    println(@sprintf("🏋️  Train: loss = %.4f, acc = %.4f", train_loss, train_acc))
    println(@sprintf("🧪  Test : loss = %.4f, acc = %.4f\n", test_loss, test_acc))
end

show(TO)

