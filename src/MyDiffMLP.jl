module MyDiffMLP

include("MyAD.jl")
include("MyNN.jl")
include("data_prep.jl")

using .MyAD
using .MyNN
using .PrepareData  

export MyAD, MyNN, prepare_dataset 

end
