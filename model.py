import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

class Net(nn.Module):
    def __init__(self, channel_use, block_size, snr, use_cuda=True, lpf_num_taps=100, drop_rate=0):
        super(Net, self).__init__()
        self.channel_use = channel_use
        self.block_size = block_size
        self.snr = snr
        self.use_cuda = use_cuda
        self.encoder = nn.Sequential(
            nn.Linear(block_size, block_size),
            nn.PReLU(),
            nn.BatchNorm1d(block_size),
            nn.Dropout(p=drop_rate),
            nn.Linear(block_size, block_size),
            nn.PReLU(),
            nn.BatchNorm1d(block_size),
            nn.Dropout(p=drop_rate),
            nn.Linear(block_size, channel_use),
        )

        dec_hidden_dim = lpf_num_taps + channel_use - 1
        self.decoder = nn.Sequential(
            nn.Linear(dec_hidden_dim, block_size),
            nn.PReLU(),
            nn.BatchNorm1d(block_size),
            nn.Dropout(p=drop_rate),
            nn.Linear(block_size, block_size),
            nn.PReLU(),
            nn.BatchNorm1d(block_size),
            nn.Dropout(p=drop_rate),
            nn.Linear(block_size, block_size)
        )
        self.conv1 = nn.Conv1d(1, 1, lpf_num_taps, padding=lpf_num_taps-1, bias=False)
        self.sig = nn.Sigmoid()

    def decode(self, x):
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)

        # Normalization so that every example x is normalized. Since bit energy should be 1, we multiply by length of x.
        x = self.block_size * (x / x.norm(dim=-1)[:, None])
        return x

    def lpf(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        return x

    def awgn(self, x):
        # Simulated Gaussian noise.
        training_signal_noise_ratio = 10**(0.1*self.snr)

        # bit / channel_use
        communication_rate = self.block_size / self.channel_use

        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        if self.use_cuda: noise = noise.cuda()
        x += noise
        return x

    def forward(self, x):
        x = self.encode(x)
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
