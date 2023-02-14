import torch
import torch.nn as nn
from animal_data_loaders import build_data_loaders

# RNN: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
print(output)

print("\n\n\n\n")

# LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output)
