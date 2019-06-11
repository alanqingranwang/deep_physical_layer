import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from Dataset import Dataset
from torch.optim import Adam, RMSprop
import math
from torch.autograd import Variable
from radio_transformer_networks import Net, CHANNEL_USE, BLOCK_SIZE, NUM_TAPS

snr = 1
model = Net(2**BLOCK_SIZE, compressed_dim=CHANNEL_USE, snr=snr, num_taps=NUM_TAPS)
model.load_state_dict(torch.load('./models/model_with_conv'+str(snr)))

for name, param in model.named_parameters():
    if name == 'conv1.weight':
        p = param.data.numpy()
        print(type(p))
        print(p.shape)
        p = p.reshape(-1)
        plt.plot(p)
        plt.savefig('filter_coeff.png')
