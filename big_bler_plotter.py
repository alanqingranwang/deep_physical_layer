'''
6/27/19
This script is for testing the big fat model with twice the layers.
Saved models are called cont_big_model and disc_big_model.
Its a (32, 7) autoencoder with dropout rate 0, trained for 1000 epochs.
'''
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from experimental import Net, get_args
import scipy.signal as signal

USE_CUDA = torch.cuda.is_available()

N = 10000
# mean_bers = np.loadtxt('mean_bers.txt')
berawgn_bers = np.loadtxt('awgn_bers.txt')
snrs = np.arange(-5, 11)

test_data = torch.randint(2, (10000, 4)).float()
test_zeros = torch.zeros(10000, 4*2).float()
test_zeros[:, :4] = test_data
test_data = test_zeros
test_labels = test_data

if USE_CUDA:
    test_data = test_data.cuda()
    test_labels = test_labels.cuda()

cont = []
disc = []
for snr in snrs:
    model = Net(channel_use=32, block_size=4, snr=snr, use_cuda=USE_CUDA, use_lpf=False, use_complex=True, dropout_rate=0)
    model.load_state_dict(torch.load('./models/with_batch_norm_' + str(snr) + '.0'))
    model.eval()
    if USE_CUDA: model = model.cuda()
    test_out = model(test_data)
    pred = torch.round(test_out)
    # print(model.accuracy(pred, test_labels))
    cont.append(1-model.accuracy(pred, test_labels))

print(cont)
# plt.semilogy(snrs, mean_bers_skip, ls = '-', color = 'b')
plt.semilogy(np.linspace(-5, 10, num=20), berawgn_bers, ls = '-', color = 'g')
plt.semilogy(snrs, cont, ls = '--', color = 'r')

legend_strings = []
# legend_strings.append('Experimental QPSK with Pulseshaping')
legend_strings.append('Theoretical QPSK')
legend_strings.append('Autoencoder with complex representation')
plt.xlabel('SNR [dB]')
plt.ylabel('Bit Error Rate')
plt.legend(legend_strings, loc = 'lower left')
plt.title('QPSK and Autoencoder')
plt.savefig('./complex.png')
