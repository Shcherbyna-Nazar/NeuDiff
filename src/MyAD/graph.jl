using Base.Threads: @threads
using LinearAlgebra: mul!


# ---- Topological Sort ----

"""
    visit_children(node, visited, order)

Visits the children of a node for topological sorting.
"""
function visit_children(node::ScalarOperator, visited::Set{GraphNode}, order::Vector{GraphNode})
    for x in node.inputs
        visit(x, visited, order)
    end
end

function visit_children(node::MatMulOperator, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.A, visited, order)
    visit(node.B, visited, order)
end

function visit_children(node::BroadcastedOperator, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.input, visited, order)
end

function visit_children(node::Conv1DOp, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.W, visited, order)
    node.b !== nothing && visit(node.b, visited, order)
    visit(node.input, visited, order)
end

function visit_children(node::MaxPool1DOp, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.x, visited, order)
end

function visit_children(node::PermuteDimsOp, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.x, visited, order)
end

function visit_children(node::FlattenOp, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.x, visited, order)
end

function visit_children(node::EmbeddingOp, visited::Set{GraphNode}, order::Vector{GraphNode})
    visit(node.weight, visited, order)
end

function visit_children(::GraphNode, ::Set{GraphNode}, ::Vector{GraphNode})
    # fallback: no children
end

function visit(node::GraphNode, visited::Set, order::Vector)
    if node ∉ visited
        push!(visited, node)
        visit_children(node, visited, order)
        push!(order, node)
    end
end

function topological_sort(root::GraphNode)
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit(root, visited, order)
    return order
end

# ---- Forward Pass ----
function forward!(nodes::Vector{GraphNode})
    for node in nodes
        forward(node)
    end
end

function forward(node::ScalarOperator)
    in1, in2 = node.inputs[1].output, node.inputs[2].output
    if node.output === nothing || size(node.output) != size(in1)
        node.output = similar(in1)
    end
    @inbounds broadcast!(node.f, node.output, in1, in2)
end

function forward(node::MatMulOperator)
    A, B = node.A.output, node.B.output
    szA, szB = size(A,1), size(B,2)
    if node.output === nothing || size(node.output) != (szA, szB)
        node.output = zeros(eltype(A), szA, szB) 
    end
    mul!(node.output, A, B)
end

function forward(node::BroadcastedOperator)
    x = node.input.output
    if node.output === nothing || size(node.output) != size(x)
        node.output = similar(x)
    end
    @inbounds broadcast!(node.f, node.output, x)
end

forward(::Constant) = nothing
forward(::Variable) = nothing

function forward(node::FlattenOp)
    orig_shape = size(node.x.output)
    flat_rows = prod(orig_shape[1:end-1])
    flat_cols = orig_shape[end]
    if node.output === nothing || size(node.output) != (flat_rows, flat_cols)
        node.output = reshape(node.x.output, flat_rows, flat_cols)
    else
        reshape!(node.output, node.x.output, flat_rows, flat_cols)
    end
    node.orig_shape = orig_shape
end

function forward(node::Conv1DOp{T}) where {T}
    x, W, b = node.input.output, node.W.output, node.b === nothing ? nothing : node.b.output
    K, C, O = size(W)
    L, _, B = size(x)
    P, S = node.padding, node.stride
    L_padded = L + 2P
    L_out = (L_padded - K) ÷ S + 1

    x_padded = node.x_padded
    if x_padded === nothing || size(x_padded) != (L_padded, C, B)
        node.x_padded = zeros(T, L_padded, C, B)
        x_padded = node.x_padded
    else
        fill!(x_padded, 0)
    end
    @views x_padded[P+1:P+L, :, :] .= x

    X_col = node.X_col
    if X_col === nothing || size(X_col) != (K*C, L_out*B)
        node.X_col = zeros(T, K*C, L_out*B)
        X_col = node.X_col
    else
        fill!(X_col, 0)
    end

    col = 1
    @inbounds for b in 1:B
        for l in 0:L_out-1
            patch = @view x_padded[l*S+1:l*S+K, :, b]
            @views X_col[:, col] .= vec(patch)
            col += 1
        end
    end

    if node.W_mat === nothing || size(node.W_mat) != (K*C, O)
        Wf = W[K:-1:1, :, :]
        node.W_mat = reshape(Wf, K*C, O)
        node.W_mat_T = transpose(node.W_mat)
    else
        Wf = W[K:-1:1, :, :]
        reshape!(node.W_mat, Wf, K*C, O)
        node.W_mat_T = transpose(node.W_mat)
    end
    W_mat_T = node.W_mat_T

    out_mat = node.out_mat
    if out_mat === nothing || size(out_mat) != (O, L_out*B)
        node.out_mat = zeros(T, O, L_out*B)
        out_mat = node.out_mat
    else
        fill!(out_mat, 0)
    end
    mul!(out_mat, W_mat_T, X_col)

    if b !== nothing
        @inbounds @views out_mat .+= reshape(b, O, 1)
    end

    out = reshape(out_mat, O, L_out, B)
    out = permutedims(out, (2, 1, 3))  

    node.output = node.activation === identity_fn ? out : node.activation.(out)
    return nothing
end

function forward(node::MaxPool1DOp)
    x = node.x.output
    L, C, B = size(x)
    k, s = node.kernel_size, node.stride
    out_len = div(L - k, s) + 1

    if node.output === nothing || size(node.output) != (out_len, C, B)
        node.output = zeros(eltype(x), out_len, C, B)
    else
        fill!(node.output, 0)
    end
    if node.indices === nothing || size(node.indices) != (out_len, C, B)
        node.indices = zeros(Int, out_len, C, B)
    else
        fill!(node.indices, 0)
    end
    out, idx = node.output, node.indices

    @threads for b in 1:B
        for c in 1:C
            @inbounds for i in 0:out_len-1
                r = i*s+1:i*s+k
                win = @view x[r, c, b]
                max_val, max_idx = findmax(win)
                out[i+1, c, b] = max_val
                idx[i+1, c, b] = r.start + max_idx - 1
            end
        end
    end
end

function forward(node::PermuteDimsOp)
    x = node.x.output
    out = permutedims(x, node.dims)
    if node.output === nothing || size(node.output) != size(out)
        node.output = similar(out)
    end
    copyto!(node.output, out)
end

function forward(node::EmbeddingOp)
    node.output = reshape(node.weight.output[:, node.indices], node.shape)
end

forward(node::GraphNode) = error("No forward method for type $(typeof(node))")
