include("src/MyAD.jl")
include("src/MyNN.jl")

using .MyAD
using .MyNN
using JLD2, Printf, Statistics, Random

function normalize(X)
    μ = mean(X, dims=2)
    σ = std(X, dims=2) .+ 1e-8
    return (X .- μ) ./ σ
end

X_train = load("data/imdb_dataset_prepared.jld2", "X_train")
y_train = load("data/imdb_dataset_prepared.jld2", "y_train")
X_test  = load("data/imdb_dataset_prepared.jld2", "X_test")
y_test  = load("data/imdb_dataset_prepared.jld2", "y_test")

function create_batches(X, Y; batchsize=64, shuffle=true)
    idxs = collect(1:size(X, 2))
    if shuffle
        Random.shuffle!(idxs)
    end
    return [(X[:, idxs[i:min(i+batchsize-1, end)]],
             Y[:, idxs[i:min(i+batchsize-1, end)]])
             for i in 1:batchsize:length(idxs)]
end

model = Chain(
    Dense(size(X_train, 1), 32, relu),
    Dense(32, 1, sigmoid)
)

function bce(ŷ, y)
    ϵ = 1e-7
    ŷ_clipped = clamp.(ŷ, ϵ, 1 .- ϵ)
    return -mean(y .* log.(ŷ_clipped) .+ (1 .- y) .* log.(1 .- ŷ_clipped))
end

function bce_grad(ŷ, y)
    ϵ = 1e-7
    return (ŷ .- y) ./ (clamp.(ŷ .* (1 .- ŷ), ϵ, 1.0)) ./ size(y, 2)
end

accuracy(ŷ, y) = mean((ŷ .> 0.5) .== (y .> 0.5))

epochs = 5
batchsize = 64
η = 0.001  # typowa wartość dla Adam

params = parameters(model)
state = AdamState(params)

for epoch in 1:epochs
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    batches = create_batches(X_train, y_train; batchsize=batchsize)

    t = @elapsed begin
        for (x, y) in batches
            x_node = Variable(x, zeros(size(x)))
            y_node = Variable(y, zeros(size(y)))

            out = model(x_node)
            graph = topological_sort(out)
            forward!(graph)

            ŷ = out.output
            l = bce(ŷ, y)
            total_loss += l
            total_acc += accuracy(ŷ, y)
            num_batches += 1

            out.gradient = bce_grad(ŷ, y)

            zero_gradients!(model)
            backward!(graph, out.gradient)
            update_adam!(state, params, η)

            if epoch == 1 && num_batches == 1
                println("🔍 Przykładowe predykcje: ", round.(ŷ[1, 1:10]; digits=4))
                @show mean(abs.(params[1].output))
                @show maximum(abs.(params[1].gradient))
            end
        end
    end

    train_loss = total_loss / num_batches
    train_acc = total_acc / num_batches

    # --- Ewaluacja ---
    x_eval = Variable(X_test, zeros(size(X_test)))
    out_eval = model(x_eval)
    forward!(topological_sort(out_eval))
    test_pred = out_eval.output
    test_loss = bce(test_pred, y_test)
    test_acc = accuracy(test_pred, y_test)

    println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.2f, a: %.2f) \tTest: (l: %.2f, a: %.2f)",
        epoch, t, train_loss, train_acc, test_loss, test_acc))
end
