import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from ComplexLinear import ComplexLinear

class Net(nn.Module):
    def __init__(self, channel_use, block_size, snr, use_cuda=True, use_lpf=True, use_complex=False, lpf_num_taps=100, dropout_rate=0):
        super(Net, self).__init__()
        self.channel_use = channel_use
        self.block_size = block_size
        self.snr = snr
        self.use_cuda = use_cuda
        self.use_lpf = use_lpf
        self.use_complex = use_complex

        self.enc_linear1 = nn.Linear(block_size, block_size)
        self.enc_linear2 = nn.Linear(block_size, channel_use)
        self.enc_clinear1 = ComplexLinear(block_size, block_size)
        self.enc_clinear2 = ComplexLinear(block_size, channel_use)

        if use_lpf:
            dec_hidden_dim = lpf_num_taps + channel_use - 1
        else:
            dec_hidden_dim = channel_use

        self.dec_linear1 = nn.Linear(dec_hidden_dim, block_size)
        self.dec_linear2 = nn.Linear(block_size, block_size)
        self.dec_clinear1 = ComplexLinear(dec_hidden_dim, block_size)
        self.dec_clinear2 = ComplexLinear(block_size, block_size)

        self.batchnorm = nn.BatchNorm1d(block_size)
        self.cbatchnorm = nn.BatchNorm1d(block_size*2)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv1 = nn.Conv1d(1, 1, lpf_num_taps, padding=lpf_num_taps-1, bias=False)
        self.prelu = nn.PReLU()
        self.sig = nn.Sigmoid()

    def encode(self, x):
        if self.use_complex:
            x = self.enc_clinear1(x)
            x = self.prelu(x)
            #x = self.cbatchnorm(x)
            x = self.dropout(x)
            x = self.enc_clinear2(x)
        else:
            x = self.enc_linear1(x)
            x = self.prelu(x)
            x = self.batchnorm(x)
            x = self.dropout(x)
            x = self.enc_linear2(x)

        return x

    def normalization(self, x):
        # Normalization so that every example x is normalized.
        # Since sample energy should be 1, we multiply by sqrt
        # of channel_use, since signal energy and vector norm are off by sqrt.
        x = self.channel_use**0.5 * (x / x.norm(dim=1)[:, None])
        return x

    def decode(self, x):
        if self.use_complex:
            x = self.dec_clinear1(x)
            x = self.prelu(x)
            #x = self.cbatchnorm(x)
            x = self.dropout(x)
            x = self.dec_clinear2(x)
        else:
            x = self.dec_linear1(x)
            x = self.prelu(x)
            x = self.batchnorm(x)
            x = self.dropout(x)
            x = self.dec_linear2(x)

        return x

    def lpf(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        return x

    def awgn(self, x):
        snr_lin = 10**(0.1*self.snr)
        rate = self.block_size / self.channel_use

        noise = torch.randn(*x.size()) * np.sqrt(1/(2 * rate * snr_lin))
        if self.use_cuda: noise = noise.cuda()
        x += noise
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.normalization(x)
        if self.use_lpf:
            x = self.lpf(x)
        x = self.awgn(x)
        x = self.decode(x)
        x = self.sig(x) # Sigmoid for BCELoss
        return x

    @staticmethod
    def accuracy(preds, labels):
        # block-wise accuracy
        acc1 = torch.sum((torch.sum(torch.abs(preds-labels), 1)==0)).item()/(list(preds.size())[0])
        # bit-wise accuracy
        acc2 = 1 - torch.sum(torch.abs(preds-labels)).item() / (list(preds.size())[0]*list(preds.size())[1])
        return acc2
