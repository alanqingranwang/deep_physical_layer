'''
Alan Wang, AL29162
6/27/19

This script contains some skeleton code to produce ber plots,
given saved models of the autoencoder.

Plots can be created per-snr or for one snr.
'''
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from main import Autoencoder

USE_CUDA = torch.cuda.is_available()

snrs = np.linspace(-5, 10, num=10)
channel_use = 7
block_size = 4
use_complex = True

_, _, test_data, test_labels = generate_data(block_size, use_complex)
if USE_CUDA:
    test_data = test_data.cuda()
    test_labels = test_labels.cuda()

cont = []
for snr in snrs:
    model = Autoencoder(channel_use=channel_use, block_size=block_size, snr=snr)
    model.load_state_dict(torch.load('./models/cont_' + str(snr)))
    model.eval()
    if USE_CUDA: model = model.cuda()
    test_out = model(test_data)
    pred = torch.round(test_out)
    cont.append(1-model.accuracy(pred, test_labels))

plt.semilogy(snrs, cont, ls = '--', color = 'r')
plt.semilogy(snrs, disc, ls = '--', color = 'k')

legend_strings = []
legend_strings.append('Autoencoder')
plt.xlabel('Eb/No')
plt.ylabel('Block Error Ratio')
plt.legend(legend_strings, loc = 'lower left')
plt.title('Autoencoder BER')
plt.savefig('./ber_plt_best_val.png')
