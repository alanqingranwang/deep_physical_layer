import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from radio_transformer_networks import Net

BATCH_SIZE = 256
CHANNEL_USE = 7
BLOCK_SIZE = 4
USE_CUDA = True 


bpsk_from_matlab = [0.1861, 0.1584, 0.1306, 0.1038, 0.0786, 0.0563, 0.0375, 0.0229, 0.0125, 0.0060, 0.0024, 0.0008, 0.0002, 0.0000]
snrs = np.linspace(-4, 9, num=14)

def accuracy(preds, labels):
    print(pred.size())
    return torch.sum(torch.eq(pred, labels)).item()/(list(preds.size())[0])

bler_autoenc = []
test_labels = (torch.rand(1500) * (2**BLOCK_SIZE)).long()
test_data = torch.eye(2**BLOCK_SIZE).index_select(dim=0, index=test_labels)
if USE_CUDA: 
    test_data = test_data.cuda()
    test_labels = test_labels.cuda()

for snr in snrs:
    model = Net(2**BLOCK_SIZE, compressed_dim=CHANNEL_USE, snr=snr)
    model.load_state_dict(torch.load('./models/model_state_'+str(snr)))
    if USE_CUDA: model = model.cuda()
    test_out = model(test_data)
    pred = torch.argmax(test_out, dim=1)
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
