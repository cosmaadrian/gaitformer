import torch
import torch.nn as nn


class Fusion(nn.Module):
    def __init__(self, input_dim, attr_dim):
        super(Fusion, self).__init__()
        self.linear = nn.Linear(input_dim+attr_dim*2, input_dim)
        self.attr_dim = attr_dim

    def forward(self, input_1, attrs):
        bs, seq = input_1.size()[:2]
        input = torch.cat([input_1, *[attr.unsqueeze(1).expand(bs,seq,self.attr_dim) for attr in attrs]], dim=-1)
        output = torch.tanh(self.linear(input))
        return output