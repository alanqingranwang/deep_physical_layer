import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from extra import Net, accuracy, CHANNEL_USE, BLOCK_SIZE, USE_CUDA, NUM_TAPS
import scipy.signal as signal

# bpsk_from_matlab = [0.1861, 0.1584, 0.1306, 0.1038, 0.0786, 0.0563, 0.0375, 0.0229, 0.0125, 0.0060, 0.0024, 0.0008, 0.0002, 0.0000]
mean_bers = np.loadtxt('mean_bers.txt')
berawgn_bers = np.loadtxt('awgn_bers.txt')
snrs = np.linspace(-5, 10, num=20)

# bler_autoenc = []
# test_data = torch.randint(2, (10000, BLOCK_SIZE)).float()
# test_labels = test_data
# if USE_CUDA:
#     test_data = test_data.cuda()
#     test_labels = test_labels.cuda()

# for snr in snrs:
#     compressed_dim = NUM_TAPS + CHANNEL_USE - 1
#     model = Net(in_channels=BLOCK_SIZE, enc_compressed_dim=CHANNEL_USE, dec_compressed_dim=compressed_dim, lpf_num_taps=NUM_TAPS, snr=7)
#     model.load_state_dict(torch.load('./models/model_lpf_shift_' + str(snr)))
#     model.eval()
#     if USE_CUDA: model = model.cuda()
#     test_out = model(test_data)
#     pred = torch.round(test_out)
#     print(accuracy(pred, test_labels))
#     bler_autoenc.append(1-accuracy(pred, test_labels))

bler_autoenc = [0.19135000000000002, 0.16894999999999993, 0.15187500000000004, 0.14487499999999998, 0.12974999999999984, 0.11457499999999998, 0.09182499999999999, 0.08404999999999996, 0.06242499999999997, 0.0489625000000000004, 0.03747499999999995, 0.02517499999999998, 0.015575000000000006, 0.014024999999999954, 0.014074999999999949, 0.015225000000000044, 0.014950000000000019, 0.014225000000000043, 0.015275000000000039, 0.017125000000000057]
print(bler_autoenc)
plt.semilogy(snrs, mean_bers, ls = '-', color = 'b')
plt.semilogy(snrs, berawgn_bers, ls = '-', color = 'g')
plt.semilogy(snrs, bler_autoenc, ls = '--', color = 'r')

legend_strings = []
legend_strings.append('Experimental QPSK with Pulseshaping')
legend_strings.append('Theoretical QPSK')
legend_strings.append('Autoencoder')
plt.xlabel('SNR [dB]')
plt.ylabel('Block Error Ratio')
plt.legend(legend_strings, loc = 'lower left')
plt.title('BLER vs SNR for QPSK and Autoencoder Schemes')
plt.savefig('./bpsk_vs_autoenc_shift.png')
