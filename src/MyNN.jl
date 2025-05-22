module MyNN

using ..MyAD

export Dense, Chain, parameters, update!, Dropout, zero_gradients!, AdamState,
       update_adam!



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


function parameters(model::Chain)
    ps = MyAD.GraphNode[]
    for layer in model.layers
        if layer isa Dense
            push!(ps, layer.W, layer.b)
        end
    end
    return ps
end


function update!(params::Vector{MyAD.GraphNode}, η::Real)
    for p in params
        p.output .-= η .* p.gradient
    end
end

function zero_gradients!(model::Chain)
    for p in parameters(model)
        p.gradient .= 0.0
    end
end

mutable struct AdamState
    m::Vector{Matrix{Float64}}  # pierwszy moment (średnia gradientów)
    v::Vector{Matrix{Float64}}  # drugi moment (średnia kwadratów gradientów)
    β1::Float64
    β2::Float64
    ϵ::Float64
    t::Int
end


function AdamState(params; β1=0.9, β2=0.999, ϵ=1e-8)
    m = [zero(p.output) for p in params]
    v = [zero(p.output) for p in params]
    return AdamState(m, v, β1, β2, ϵ, 0)
end


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


end # module
