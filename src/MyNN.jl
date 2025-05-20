module MyNN

using ..MyAD

export Dense, Chain, parameters, update!, Dropout



struct Dense
    W::MyAD.Variable
    b::MyAD.Variable
    activation::Function
end


function Dense(in::Int, out::Int, act = MyAD.identity_fn)
    std = act == MyAD.relu ? sqrt(2 / in) : sqrt(6.0 / (in + out))  # He lub Xavier
    W = MyAD.Variable(randn(out, in) * std, zeros(out, in))         # UWAGA: randn, nie rand
    b = MyAD.Variable(zeros(out, 1), zeros(out, 1))
    return Dense(W, b, act)
end


function (layer::Dense)(x::MyAD.GraphNode)
    z = MyAD.MatMulOperator(layer.W, x, nothing, nothing)
    z = MyAD.ScalarOperator(broadcast_add, z, layer.b)
    return MyAD.BroadcastedOperator(layer.activation, z)
end

struct Dropout
    rate::Float64
end

function (d::Dropout)(x::MyAD.GraphNode)
    return x  # dropout jest ignorowany w inferencji; za chwilę dodamy wersję treningową
end



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

"""
    parameters(model::Chain)

Zwraca wszystkie parametry uczące się (Variable) z modelu.
"""
function parameters(model::Chain)
    ps = MyAD.GraphNode[]
    for layer in model.layers
        if layer isa Dense
            push!(ps, layer.W, layer.b)
        end
    end
    return ps
end


"""
    update!(params, η)

Aktualizuje wartości parametrów przez odejmowanie gradientu z krokiem η.
"""
function update!(params::Vector{MyAD.GraphNode}, η::Real)
    for p in params
        p.output .-= η .* p.gradient
    end
end

end # module
