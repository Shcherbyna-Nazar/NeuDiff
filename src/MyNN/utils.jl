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