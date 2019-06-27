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

mean_bers = np.loadtxt('mean_bers.txt')
berawgn_bers = np.loadtxt('awgn_bers.txt')
snrs = np.linspace(-5, 10, num=10)

args = get_args()

test_data = torch.tensor(np.loadtxt('./test_data.txt')).float()
test_labels = test_data
if USE_CUDA:
    test_data = test_data.cuda()
    test_labels = test_labels.cuda()

cont = []
disc = []
for snr in snrs:
    com_dim = args.lpf_taps + args.channel_use - 1
    model = Net(channel_use=args.channel_use, block_size=args.block_size, snr=snr)
    model.load_state_dict(torch.load('./models/cont_big_model_' + str(snr)))
    model.eval()
    if USE_CUDA: model = model.cuda()
    test_out = model(test_data)
    pred = torch.round(test_out)
    print(model.accuracy(pred, test_labels))
    cont.append(1-model.accuracy(pred, test_labels))

for snr in snrs:
    com_dim = args.lpf_taps + args.channel_use - 1
    model = Net(channel_use=args.channel_use, block_size=args.block_size, snr=snr)
    model.load_state_dict(torch.load('./models/disc_big_model_' + str(snr)))
    model.eval()
    if USE_CUDA: model = model.cuda()
    test_out = model(test_data)
    pred = torch.round(test_out)
    print(model.accuracy(pred, test_labels))
    disc.append(1-model.accuracy(pred, test_labels))

mean_bers_skip = mean_bers[::2]
berawgn_bers_skip = berawgn_bers[::2]
plt.semilogy(snrs, mean_bers_skip, ls = '-', color = 'b')
plt.semilogy(snrs, berawgn_bers_skip, ls = '-', color = 'g')
plt.semilogy(snrs, cont, ls = '--', color = 'r')
plt.semilogy(snrs, disc, ls = '--', color = 'k')

legend_strings = []
legend_strings.append('Experimental QPSK with Pulseshaping')
legend_strings.append('Theoretical QPSK')
legend_strings.append('Autoencoder with continuous')
legend_strings.append('Autoencoder with discrete')
plt.xlabel('SNR [dB]')
plt.ylabel('Block Error Ratio')
plt.legend(legend_strings, loc = 'lower left')
plt.title('QPSK and Autoencoder')
plt.savefig('./ber_plt_best_val.png')
