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
from sklearn.manifold import TSNE
import itertools

USE_CUDA = torch.cuda.is_available()

channel_use = 64
block_size = 64
default_snr = 8
use_complex = True

# model = Net(channel_use=channel_use, block_size=block_size, snr=default_snr, use_cuda=USE_CUDA, use_lpf=False, use_complex=True, dropout_rate=0)
# model.load_state_dict(torch.load('./models/complex_(%s,%s)_%s.0' % (channel_use, block_size, default_snr)))
# model = model.cuda()

# lst = torch.tensor(list(map(list, itertools.product([0, 1], repeat=block_size))))
# sample_data = torch.zeros((len(lst), block_size*2))
# sample_data[:, :block_size] = lst
# train_codes = model.encode(sample_data.cuda()).cuda()
# train_codes_cpu = train_codes.cpu().detach().numpy()
# embed = TSNE().fit_transform(train_codes_cpu)
# print(embed.shape)

# plt.scatter(embed[:,0], embed[:,1])
# plt.title('t-SNE Embedding of (%s, %s) Encoding' % (channel_use, block_size))
# plt.savefig('./tsne.png')

N = 100000
# mean_bers = np.loadtxt('mean_bers.txt')
berawgn_bers = np.loadtxt('awgn_bers.txt')
bch_bers = np.loadtxt('bch_bers.txt')
snrs = np.arange(-5, 11)

test_real_data = torch.randint(2, (N, block_size)).float().cuda()

test_complex_data = torch.zeros(N, block_size*2).float().cuda()
test_complex_data[:, :block_size] = test_real_data

test_real_labels = test_real_data
test_complex_labels = test_complex_data

if USE_CUDA:
    test_real_data = test_real_data.cuda()
    test_real_labels = test_real_labels.cuda()
    test_complex_data = test_complex_data.cuda()
    test_complex_labels = test_complex_labels.cuda()

bigmodel = []
for snr in snrs:
    print(snr)
    model = Net(channel_use=channel_use, block_size=block_size, snr=snr, use_cuda=USE_CUDA, use_lpf=False, use_complex=True, dropout_rate=0)
    model.load_state_dict(torch.load('./models/complex_(64,64)_' + str(8) + '.0'))
    model.eval()
    if USE_CUDA: model = model.cuda()
    test_out = model(test_complex_data)
    pred = torch.round(torch.sigmoid(test_out))
    # print(model.accuracy(pred, test_labels))
    bigmodel.append(1-model.accuracy(pred, test_complex_labels))

# plt.semilogy(snrs, mean_bers_skip, ls = '-', color = 'b')

plt.semilogy(np.linspace(-5, 10, num=20), berawgn_bers, ls = '-', color = 'g')
plt.semilogy(snrs, bch_bers, ls = '--', color = 'r')
plt.semilogy(snrs, bigmodel, ls = '--', color = 'b')

legend_strings = []
legend_strings.append('Theoretical QPSK')
legend_strings.append('(128, 64) BCH and QPSK')
# legend_strings.append('(4,4) Autoencoder, small')
legend_strings.append('(64,64) Autoencoder')
plt.xlabel('Eb/No')
plt.ylabel('Bit Error Rate')
plt.legend(legend_strings, loc = 'lower left')
plt.title('BCH/QPSK and (%s, %s) Autoencoder' % (str(channel_use), str(block_size)))
plt.savefig('./bch.png')
