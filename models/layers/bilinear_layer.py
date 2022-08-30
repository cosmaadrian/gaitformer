import math
import torch
import torch.nn as nn
import torch.nn.init as init

class Bilinear(nn.Module):
    def __init__(self, input1_dim, input2_dim, bias=True):
        super(Bilinear, self).__init__()
        self.bilinear_weights = nn.Parameter(torch.rand(input1_dim, input2_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input2_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def forward(self, input_1, input2):
        x = torch.matmul(input_1, self.bilinear_weights)
        output = torch.mul(x, input2.unsqueeze(1)) # (bs, time_step, dim) * (bs, 1, dim)
        if self.bias is not None:
            output += self.bias
        return output

    def reset_parameters(self):
        init.kaiming_uniform_(self.bilinear_weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.bilinear_weights)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)