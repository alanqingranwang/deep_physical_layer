"""
Alan Wang, AL29162
7/31/19

Neural network layers for handling complex numbers.
Represent complex numbers as [1, 2*N] vector, where
real part is in [:N] and imag part is in [N:]
"""

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
        real = self.lin_real(x[:,:self.in_dim]) - self.lin_imag(x[:,self.in_dim:])
        imag = self.lin_real(x[:,self.in_dim:]) + self.lin_imag(x[:,:self.in_dim])
        output = torch.cat((real, imag), dim=-1)
        return output

class ComplexConv(nn.Module):
    def __init__(self, num_taps, padding):
        super(ComplexConv, self).__init__()
        self.conv_real = nn.Conv1d(1, 1, num_taps, padding=padding, bias=False)
        self.conv_imag = nn.Conv1d(1, 1, num_taps, padding=padding, bias=False)

    def forward(self, x):
        input_dim = x.shape[-1] // 2
        real = self.conv_real(x[:,:,:input_dim]) - self.conv_imag(x[:,:,input_dim:])
        imag = self.conv_real(x[:,:,input_dim:]) + self.conv_imag(x[:,:,:input_dim])
        output = torch.cat((real, imag), dim=-1)
        return output
