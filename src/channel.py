import torch
import numpy as np

class Channel():
    def __init__(self, snr, block_size, channel_use, use_cuda):
        self.snr = snr
        self.block_size = block_size
        self.channel_use = channel_use
        self.use_cuda = use_cuda

    def awgn(self, x):
        snr_lin = 10**(0.1*self.snr)
        rate = self.block_size / self.channel_use

        noise = torch.randn(*x.size()) * (1/(2 * rate * snr_lin))**0.5
        if self.use_cuda: noise = noise.cuda()
        x = x + noise
        return x

    def time_varying_tones(self, x, discrete_jumps, num_samples, signal_length):
        L = np.linspace(0, np.pi, num=discrete_jumps)
        snr_lin = 10**(0.1*self.snr)
        rate = self.block_size / self.channel_use

        noise = torch.randn(*x.size()) * np.sqrt(1/(2 * rate * snr_lin))
        noise = noise.squeeze() # Unstable probably
        noise_cpu = noise.detach().numpy()

        # Convolve each sequence element by a sin function that rotates in frequency
        # at increment given by discrete_jumps
        res = torch.zeros(x.shape)
        for i in range(noise_cpu.shape[0]):
            freq = i % discrete_jumps
            omega = L[freq]
            t = np.linspace(0, 63, num=signal_length)
            noise_tone = torch.tensor(np.sin(omega * t)).float()

            noise_convolved = np.convolve(noise_tone, noise_cpu[i], 'same')
            noise_convolved = torch.tensor(noise_convolved).float()
            res[i] = noise_convolved.view(1, -1)

        if self.use_cuda:
            res = res.cuda()
        x = x + res
        return x
