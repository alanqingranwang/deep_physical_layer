import numpy as np
from scipy.signal import firwin
import torch

def channel(codes, channel_use, snr, cuda):
    with torch.no_grad():
        codes = codes.detach().numpy()
        lpf = firwin(channel_use, 0.5)
        res = np.zeros(codes.shape)
        for i, sig in enumerate(codes):
            res[i] = np.convolve(sig, lpf, 'same')
        return torch.tensor(res, requires_grad=True).float()

    # snr_lin = 10**(0.1*snr)
    # rate = 1
    # noise = torch.randn(*codes.size()) *np.sqrt(1/(2 * rate * snr_lin))
    # if cuda:
    #     noise = noise.cuda()
    # codes += noise
    # return codes
