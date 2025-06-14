using PyCall
using BenchmarkTools

torch = pyimport("torch")
nn = pyimport("torch.nn")
F = pyimport("torch.nn.functional")

# Model hyperparameters
vocab_size, emb_dim, seq_len, batch_size = 3000, 32, 50, 16
conv_out_channels = 64
kernel_size = 3
pool_kernel, pool_stride = 2, 2
flattened_dim = ((seq_len - kernel_size + 1) ÷ pool_stride) * conv_out_channels
hidden_dim = 64
output_dim = 8

# Create the PyTorch model in Python
py"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyTorchModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, conv_out_channels, kernel_size, flattened_dim, hidden_dim, output_dim, pool_kernel, pool_stride):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.conv = nn.Conv1d(emb_dim, conv_out_channels, kernel_size)
        self.pool = nn.MaxPool1d(pool_kernel, pool_stride)
        self.fc1 = nn.Linear(flattened_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embed(x)           # (batch, seq_len, emb_dim)
        x = x.permute(0,2,1)        # (batch, emb_dim, seq_len)
        x = self.conv(x)            # (batch, out_ch, L_out)
        x = F.relu(x)
        x = self.pool(x)            # (batch, out_ch, new_L)
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import __main__
__main__.pt_model = MyTorchModel(
    $vocab_size, $emb_dim, $conv_out_channels, $kernel_size,
    $flattened_dim, $hidden_dim, $output_dim, $pool_kernel, $pool_stride
)
"""

pt_model = pyimport("__main__").pt_model

# Prepare random data (seq_len, batch_size) => (batch_size, seq_len)
indices = rand(1:vocab_size, seq_len, batch_size)
indices_pt = torch.tensor(permutedims(indices, (2,1)), dtype=torch.long)  # (batch, seq_len)

y_true = randn(Float32, output_dim, batch_size)
y_true_pt = torch.tensor(permutedims(y_true, (2,1)), dtype=torch.float32)  # (batch, output_dim)

function pytorch_fullpass()
    pt_model.zero_grad()
    out = pt_model(indices_pt)
    loss = F.mse_loss(out, y_true_pt)
    loss.backward()
end

println("PyTorch: Forward + Backward + Grad")
@btime pytorch_fullpass()
