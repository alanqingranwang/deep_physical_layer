import numpy as np
import scipy.signal as signal
from commpy.modulation import PSKModem
from commpy.filters import rrcosfilter
import torch
import matplotlib.pyplot as plt

import imageio

def _downsample(sig, r):
    return sig[::r]

def _upsample(sig, r):
    res = np.zeros(len(sig)*r, dtype=complex)
    for i in range(len(res)):
        if i % r == 0:
            res[i] = sig[int(i / r)]
    return res

def _convert_to_bit_stream(x, bits_per_code):
    res = []
    for e in x:
        binary = bin(e)
        b_ind = binary.index('b')
        binary = binary[b_ind+1:].zfill(bits_per_code)
        for mint in binary:
            res.append(int(mint))
    return np.array(res)

def _convert_to_int_stream(x, M):
    res = []
    for i in range(0, len(x)-M+1, M):
        binary = x[i:i+M]
        res.append(int("".join(str(x) for x in binary), 2))
    return np.array(res)

def pulseshape_lowpass(x, samps_per_symb, batch_size, h_lowpass, USE_CUDA):

    ###### Start of sketchy stuff #######
    x = x.cpu().detach().numpy()

    # We would like to learn an optimal coding for this given modulation scheme.
    # An optimal coding will be of whatever length x is, because the encoder will add redundancy
    # to the original signal s. Therefore, the value of M should be 2 to the power of the length of x, since the
    # number of total symbols to modulate is 2**len(x)

    res = []
    # for b in range(batch_size):
    #     x[b] = [int(np.round(x[b, i])) for i in range(len(x[b]))]
    # x = x.astype(int)

    for b in range(batch_size):

        x_samples = _upsample(x[b], samps_per_symb)

        # Filter with pulse shape filter first
        #x_pulseshaped = np.convolve(x_samples, h_pulseshape, 'same') # Waveform with PSF
        L = 100
        # Then low pass filter
        # if(b == 10):
        #     zp = np.zeros(L)
        #     zp[:len(x_samples)] = x_samples
        #     fig = plt.figure()
        #     X_samp = np.fft.fft(zp)
        #     plt.plot(np.abs(X_samp))
        #     plt.savefig('results/images/fft_test.png')
        #     fig.clf()
        #     plt.close()

        # if(b == 10):
        #     zp = np.zeros(L)
        #     zp[:len(h_lowpass)] = h_lowpass
        #     fig = plt.figure()
        #     H = np.fft.fft(zp)
        #     plt.plot(np.abs(H))
        #     plt.savefig('results/images/fft_lowpass.png')
        #     fig.clf()
        #     plt.close()

        x_lowpassed = np.convolve(x_samples, h_lowpass, 'full')
        # if(b == 10):
        #     zp = np.zeros(L)
        #     zp[:len(x_lowpassed)] = x_lowpassed
        #     fig = plt.figure()
        #     X_low = np.fft.fft(zp)
        #     plt.stem(np.abs(X_low))
        #     plt.savefig('results/images/fft_test1.png')
        #     fig.clf()
        #     plt.close()

        # Receive by match filtering first before downsample
        #x_lowpassed = _downsample(x_lowpassed, samps_per_symb) * samps_per_symb
        #if b == batch_size-1:
        #    plt.figure()
        #    plt.xlim([-1, 1])
        #    plt.ylim([-1, 1])
        #    #plt.scatter(y.real, y.imag, color = 'b', s = 1)
        #    plt.scatter(x_mod.real, x_mod.imag, color = 'g')
        #    plt.title('Pulse shaped receive symbols')
        #    plt.savefig('./pulseshaped_receive_symbols')
        #    plt.close()

        # Convert back to integers
        res.append(x_lowpassed.real)

    res = torch.from_numpy(np.array(res)).float()
    if USE_CUDA: res = res.cuda()
    return res
