push!(LOAD_PATH, "src/MyAD")
push!(LOAD_PATH, "src/MyNN")

using MyAD
using MyNN
using Printf, Statistics, DelimitedFiles, Dates, Random, Plots

# === Dane: dwie spirale ===
function generate_spirals(n::Int = 150, noise::Float64 = 0.1)
    θ = range(0, 4π, length=n)
    r = θ
    x1 = vcat((r .* cos.(θ))', (r .* sin.(θ))') .+ noise .* randn(2, n)
    x2 = vcat((-r .* cos.(θ))', (-r .* sin.(θ))') .+ noise .* randn(2, n)
    X = hcat(x1, x2)                 # (2, 2n)
    Y = reshape(vcat(zeros(n), ones(n)), 1, 2n)  # (1, 2n)
    return X ./ 6.5, Y               # skalowanie wejścia
end

X, Y = generate_spirals(150, 0.1)
X = (X .- mean(X, dims=2)) ./ std(X, dims=2)

"Zamienia dane na zmienną z AD"
to_variable(x::Matrix{Float64}) = MyAD.Variable(x, zeros(size(x)))

model = Chain(
    Dense(2, 128, tanh),
    Dense(128, 128, tanh),
    Dense(128, 64, tanh),
    Dense(64, 1, sigmoid)
)


# === Metryki ===
function forward_pass(model, x)
    out = model(x)
    forward!(topological_sort(out))
    return out
end

function loss(model, x, y)
    ŷ = forward_pass(model, x).output
    ε = 1e-7
    return -mean(y.output .* log.(ŷ .+ ε) .+ (1 .- y.output) .* log.(1 .- ŷ .+ ε))
end


function accuracy(model, x, y)
    ŷ = forward_pass(model, x).output
    return mean((ŷ .> 0.5) .== (y.output .> 0.5))
end

initial_lr = 0.05
function decayed_lr(epoch)
    return initial_lr * 0.95 ^ (epoch / 1000)  # spowalniamy spadek
end


epochs = 5000

losses = Float64[]

println("📦 Trening modelu na danych Two Spirals\n")

for epoch in 1:epochs
    η = decayed_lr(epoch)
    x = to_variable(X)
    y = to_variable(Y)

    out = model(x)
    graph = topological_sort(out)
    forward!(graph)

    ŷ = out.output
    ε = 1e-7
    l = -mean(y.output .* log.(ŷ .+ ε) .+ (1 .- y.output) .* log.(1 .- ŷ .+ ε))
    acc = mean((ŷ .> 0.5) .== (y.output .> 0.5))
    push!(losses, l)

    out.gradient = (ŷ .- y.output) ./ (ŷ .* (1 .- ŷ) .+ 1e-7) ./ size(y.output, 2)  # przy CE

    backward!(graph, out.gradient)
    update!(parameters(model), η)

    if epoch % 200 == 0 || epoch == 1
        @printf("Epoch %4d │ Loss: %.5f │ Accuracy: %.2f%%\n", epoch, l, acc * 100)
    end
end

# === Ewaluacja końcowa ===
println("\n🧪 Ewaluacja końcowa:")
x = to_variable(X)
final_out = forward_pass(model, x).output

@printf("\n%-10s %-15s %-15s\n", "Sample", "Prediction", "Target")
for i in 1:size(X, 2)
    pred = round(final_out[1, i]; digits=4)
    targ = Y[1, i]
    @printf("%-10d %-15.4f %-15.1f\n", i, pred, targ)
end

# === Zapis strat ===
writedlm("spiral_training_loss.csv", losses, ',')
println("\n📉 Loss zapisany do spiral_training_loss.csv")

# === Wykres strat ===
plot(
    1:epochs, losses,
    xlabel = "Epoch",
    ylabel = "Loss",
    title = "Training Loss (Two Spirals)",
    label = "Loss",
    lw = 2,
    legend = :topright
)
savefig("spiral_loss_plot.png")
println("📊 Wykres strat zapisany jako spiral_loss_plot.png")
