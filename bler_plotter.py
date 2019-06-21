import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from extra import Net, accuracy, CHANNEL_USE, BLOCK_SIZE, SAMPS_PER_SYMB, LPF_CUTOFF, USE_CUDA

bpsk_from_matlab = [0.1861, 0.1584, 0.1306, 0.1038, 0.0786, 0.0563, 0.0375, 0.0229, 0.0125, 0.0060, 0.0024, 0.0008, 0.0002, 0.0000]
snrs = np.linspace(-4, 9, num=14)

bler_autoenc = []
train_data = torch.randint(2, (1500, BLOCK_SIZE)).float()
train_labels = train_data
if USE_CUDA:
    test_data = test_data.cuda()
    test_labels = test_labels.cuda()

h_lowpass = torch.from_numpy(signal.firwin(100, LPF_CUTOFF)).float()
for snr in snrs:
    compressed_dim = len(h_lowpass) + CHANNEL_USE*SAMPS_PER_SYMB - 1
    model = Net(in_channels=BLOCK_SIZE, enc_compressed_dim=CHANNEL_USE, dec_compressed_dim=compressed_dim, h_lowpass=h_lowpass, snr=7)
    model.load_state_dict(torch.load('./models/model_state_lpf_'+str(snr)))
    if USE_CUDA: model = model.cuda()
    test_out = model(test_data)
    pred = torch.round(test_out)
    print(accuracy(pred, test_labels))
    bler_autoenc.append(1-accuracy(pred, test_labels))

print(bler_autoenc)
plt.semilogy(snrs, bpsk_from_matlab, ls = '-', color = 'b')
plt.semilogy(snrs, bler_autoenc, ls = '--', color = 'r', marker = 'o')

legend_strings = []
legend_strings.append('Uncoded BPSK')
legend_strings.append('Autoencoder (7,4)')
plt.xlabel('SNR [dB]')
plt.ylabel('Block Error Ratio')
plt.legend(legend_strings, loc = 'lower left')
plt.title('BLER vs SNR for Autoencoder and BPSK')
plt.savefig('./sample.png')
