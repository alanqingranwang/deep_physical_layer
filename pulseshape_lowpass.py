import numpy as np
import scipy.signal as signal
from commpy.modulation import PSKModem
from commpy.filters import rrcosfilter
import torch
import matplotlib.pyplot as plt

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

def pulseshape_lowpass(x, samps_per_symb, batch_size, USE_CUDA):

        h_lowpass = np.loadtxt('h_lowpass.txt')
        h_pulseshape = np.loadtxt('h_pulseshape.txt')
        ###### Start of sketchy stuff #######
        x = x.cpu().detach().numpy()

        # We would like to learn an optimal coding for this given modulation scheme. 
        # An optimal coding will be of whatever length x is, because the encoder will add redundancy
        # to the original signal s. Therefore, the value of M should be the 2 to the power of the length of x, since the 
        # number of total symbols to modulate is 2**len(x)
        bits_per_code = x.shape[1]

        # Total number of possible codes (i.e. number of constellation points in a plot)
        M = 2**bits_per_code
        samps_per_symb = 20

        # Simple PSK modulator
        hmod = PSKModem(M)

        res = np.zeros((batch_size, x.shape[1]))
        for b in range(batch_size):
            x[b] = [int(np.round(x[b, i])) for i in range(len(x[b]))]
        x = x.astype(int)
        for b in range(batch_size):
            sign_mask = [-1 if x[b][i] < 0 else 1 for i in range(len(x[b]))]

            # Convert to bit stream
            x_bit = _convert_to_bit_stream(x[b], bits_per_code)
            #print('x_bit', x_bit.shape)

            # Modulate them
            x_mod = hmod.modulate(x_bit)
            x_samples = _upsample(x_mod, samps_per_symb);
            #print('x_samples', x_samples.shape)

            # Filter with pulse shape filter first
            x_pulseshaped = np.convolve(x_samples, h_pulseshape, 'same') # Waveform with PSF
            #print('x_pulseshaped', x_pulseshaped.shape)

            # Then low pass filter
            x_pulseshaped_lowpassed = np.convolve(x_pulseshaped, h_lowpass, 'same')
            #print('x_pulseshaped_lowpassed', x_pulseshaped_lowpassed.shape)

            # Receive by match filtering first before downsample
            y = _downsample(np.convolve(x_pulseshaped_lowpassed, h_pulseshape, 'same'), samps_per_symb) * samps_per_symb
            #print('y', y.shape)
            if b == batch_size-1:
                plt.figure()
                plt.xlim([-1, 1])
                plt.ylim([-1, 1])
                #plt.scatter(y.real, y.imag, color = 'b', s = 1)
                plt.scatter(x_mod.real, x_mod.imag, color = 'g') 
                plt.title('Pulse shaped receive symbols')
                plt.savefig('./pulseshaped_receive_symbols')
                plt.close()
            # Demodulate
            demod_bits = hmod.demodulate(y, 'hard')

            # Convert back to integers
            res_b = _convert_to_int_stream(demod_bits, bits_per_code)
            res_b = np.multiply(res_b, sign_mask)
            if(np.sum(res_b - x[b]) != 0 ):
                print('Modulation/demodulation error was ' + np.sum(res_b - x[b])) 
            res[b] = res_b

        res = torch.from_numpy(np.array(res)).float()
        if USE_CUDA: res = res.cuda()
        return res
