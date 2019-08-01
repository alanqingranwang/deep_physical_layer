import torch
import torch.nn as nn
import numpy as np
from channel import Channel

class RNN(nn.Module):
    def __init__(self, channel_use, block_size, n_layers, snr, num_taps, use_cuda, channel_type, discrete_jumps=5):
        super(RNN, self).__init__()
        self.channel_use = channel_use
        self.block_size = block_size
        self.n_layers = n_layers
        self.snr = snr
        self.num_taps = num_taps
        self.discrete_jumps = discrete_jumps
        self.use_cuda = use_cuda
        self.channel_type = channel_type

        # RNN Layer
        self.rnn = nn.RNN(block_size, channel_use, n_layers)
        # Fully connected layer
        self.fc = nn.Linear(channel_use, block_size)

    def forward(self, x):
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x)

        channel = Channel(self.snr, self.block_size, self.channel_use, self.use_cuda)
        if self.channel_type == 'awgn':
            out = channel.awgn(out)
        elif self.channel_type == 'time-varying tones':
            out = channel.time_varying_tones(out, self.discrete_jumps, num_samples=5*self.discrete_jumps, signal_length=x.shape[2])
        else:
            raise Exception('Invalid channel type')

        out = self.fc(out)

        return out, hidden

    @staticmethod
    def accuracy(preds, labels):
        preds = preds.squeeze()
        labels = labels.squeeze()
        return 1 - torch.sum(torch.abs(preds-labels)).item() / (list(preds.size())[0]*list(preds.size())[1])
