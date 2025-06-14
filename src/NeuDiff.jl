module NeuDiff

include("MyAD/MyAD.jl")
include("MyNN/MyNN.jl")

using .MyAD
using .MyNN

export MyAD, MyNN

end
