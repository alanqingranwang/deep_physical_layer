import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from ComplexLinear import ComplexLinear, ComplexConv
from scipy.signal import firwin

class Net(nn.Module):
    def __init__(self, channel_use, block_size, snr, use_cuda=True, use_lpf=True, use_complex=False, lpf_num_taps=100, dropout_rate=0):
        super(Net, self).__init__()
        self.channel_use = channel_use
        self.block_size = block_size
        self.snr = snr
        self.use_cuda = use_cuda
        self.use_lpf = use_lpf
        self.use_complex = use_complex
        self.lpf_num_taps = lpf_num_taps

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

    def decode(self, x):
        if self.use_complex:
            x = self.complex_decoder(x)
        else:
            x = self.decoder(x)
        return x

    def lpf(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        return x

    def awgn(self, x):
        f1, f2 = 0.5, 0.9
        awgn_filter = firwin(self.lpf_num_taps, f2)

        snr_lin = 10**(0.1*self.snr)
        rate = self.block_size / self.channel_use
        noise = torch.randn(*x.size()) * np.sqrt(1/(2 * rate * snr_lin))

        noise_cpu = noise.detach().numpy()
        noise_clipped = np.convolve(awgn_filter, noise_cpu[0], 'same')
        noise_clipped = torch.tensor(noise_clipped).float().cuda()
        x += noise_clipped
        return x, noise_clipped

    def calc_energy(self, x):
        x_comp = x.view(-1, 2, x.shape[1] // 2)
        x_abs = torch.norm(x_comp, dim=1)
        x_sq = torch.mul(x_abs, x_abs)
        e = torch.sum(x_sq, dim=1)
        print(e)

    def forward(self, x):
        x = self.encode(x)
        # self.calc_energy(x)
        if self.use_lpf:
            x = self.lpf(x)
        x, noise = self.awgn(x)
        # self.calc_energy(x)
        x = self.decode(x)
        return x, noise

    @staticmethod
    def accuracy(preds, labels):
        # block-wise accuracy
        acc1 = torch.sum((torch.sum(torch.abs(preds-labels), 1)==0)).item()/(list(preds.size())[0])
        # bit-wise accuracy
        acc2 = 1 - torch.sum(torch.abs(preds-labels)).item() / (list(preds.size())[0]*list(preds.size())[1])
        return acc2
