
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
