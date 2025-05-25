module MyDiffMLP

include("MyAD.jl")
include("MyNN.jl")

using .MyAD
using .MyNN

export MyAD, MyNN

end
