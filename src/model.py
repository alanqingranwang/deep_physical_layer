'''
Alan Wang, AL29162
7/31/19

Autoencoder model. Includes encoder, decoder, and channel.
'''
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from ComplexLayers import ComplexLinear, ComplexConv
from channel import channel

class Autoencoder(nn.Module):
    def __init__(self, channel_use, block_size, snr, use_cuda=True, use_lpf=True, use_complex=False, lpf_num_taps=100, dropout_rate=0):
        super(Autoencoder, self).__init__()
        self.channel_use = channel_use
        self.block_size = block_size
        self.snr = snr
        self.use_cuda = use_cuda
        self.use_lpf = use_lpf
        self.use_complex = use_complex

        if use_lpf:
            dec_hidden_dim = lpf_num_taps + channel_use - 1
        else:
            dec_hidden_dim = channel_use

        self.conv1 = ComplexConv(lpf_num_taps, padding=lpf_num_taps-1)
        self.sig = nn.Sigmoid()

        self.encoder = nn.Sequential(
                nn.Linear(block_size, block_size),
                nn.PReLU(),
                nn.BatchNorm1d(block_size),
                nn.Linear(block_size, channel_use)
                )
        self.decoder = nn.Sequential(
                nn.Linear(dec_hidden_dim, block_size),
                nn.PReLU(),
                nn.BatchNorm1d(block_size),
                nn.Linear(block_size, block_size)
                )

        self.complex_encoder = nn.Sequential(
                ComplexLinear(block_size, block_size),
                nn.PReLU(),
                ComplexLinear(block_size, channel_use)
                )
        self.complex_decoder = nn.Sequential(
                ComplexLinear(dec_hidden_dim, block_size),
                nn.PReLU(),
                ComplexLinear(block_size, block_size)
                )

    def encode(self, x):
        if self.use_complex:
            x = self.complex_encoder(x)
        else:
            x = self.encoder(x)

        # Normalization so that every example x is normalized.
        # Since sample energy should be 1, we multiply by sqrt
        # of channel_use, since signal energy and vector norm are off by sqrt.
        x = self.channel_use**0.5 * (x / x.norm(dim=1)[:, None])
        return x

    def channel(self, x):
        snr_lin = 10**(0.1*self.snr)
        rate = self.block_size / self.channel_use

        noise = torch.randn(*x.size()) * np.sqrt(1/(2 * rate * snr_lin))
        if self.use_cuda: noise = noise.cuda()
        x += noise
        return x

    def decode(self, x):
        if self.use_complex:
            x = self.complex_decoder(x)
        else:
            x = self.decoder(x)
        return x

    def forward(self, x, dynamic_snr):
        x = self.encode(x)
        x = self.channel(x)
        x = self.decode(x)
        return x

    @staticmethod
    def accuracy(preds, labels):
        # bit-wise accuracy
        return 1 - torch.sum(torch.abs(preds-labels)).item() / (list(preds.size())[0]*list(preds.size())[1])
