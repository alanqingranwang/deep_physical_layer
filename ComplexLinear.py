import torch
import torch.nn as nn
import numpy as np

class ComplexLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ComplexLinear, self).__init__()
        self.in_dim = in_dim
        self.lin_real = nn.Linear(in_dim, out_dim)
        self.lin_imag = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        input_dim = self.in_dim
        real = self.lin_real(x[:,:input_dim]) - self.lin_imag(x[:,input_dim:])
        imag = self.lin_real(x[:,input_dim:]) + self.lin_imag(x[:,:input_dim])
        output = torch.cat((real, imag), dim=1)
        return output
