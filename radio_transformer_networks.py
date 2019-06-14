import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from Dataset import Dataset
from torch.optim import Adam, RMSprop
import math
from torch.autograd import Variable

import scipy.signal as signal
from commpy.modulation import PSKModem
from commpy.filters import rrcosfilter

import imageio
from tqdm import tqdm

# Machine learning parameters
NUM_EPOCHS = 300
BATCH_SIZE = 256 
NUM_TRAINING = 10240

# Comms parameters
CHANNEL_USE = 7 # The parameter n
BLOCK_SIZE = 4 # The parameter k

# Torch parameters
USE_CUDA = True 

class Net(nn.Module):
    def __init__(self, in_channels, compressed_dim, snr):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.compressed_dim = compressed_dim
        self.snr = snr
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels, compressed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        ) 

        self.h_lowpass = np.loadtxt('h_lowpass.txt')
        self.h_pulseshape = np.loadtxt('h_pulseshape.txt')

    def decode(self, x):
        return self.decoder(x)

    def encode(self, x):
        x = self.encoder(x)

        # Normalization so that every element of x is normalized. Forces unit amplitude...? question about coding vs modulation
        #x = (x / x.norm(dim=-1)[:, None])

        # Normalization. Forces average power to be equal to n.
        x = x / x.norm() * 70

        return x

    def forward(self, x):
        batch_size = len(x)
        x = self.encode(x)

        # 7dBW to SNR.
        training_signal_noise_ratio = 10**(0.1*self.snr) 

        # bit / channel_use
        communication_rate = BLOCK_SIZE / CHANNEL_USE   


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
            x_bit = self._convert_to_bit_stream(x[b], bits_per_code, batch_size)
            #print('x_bit', x_bit.shape)

            # Modulate them
            x_mod = hmod.modulate(x_bit)
            #print('x_mod', x_mod.shape)
            x_samples = self._upsample(x_mod, samps_per_symb);
            #print('x_samples', x_samples.shape)

            # Filter with pulse shape filter first
            x_pulseshaped = np.convolve(x_samples, self.h_pulseshape, 'same') # Waveform with PSF
            #print('x_pulseshaped', x_pulseshaped.shape)

            # Then low pass filter
            x_pulseshaped_lowpassed = np.convolve(x_pulseshaped, self.h_lowpass, 'same')
            #print('x_pulseshaped_lowpassed', x_pulseshaped_lowpassed.shape)

            # Receive by match filtering first before downsample
            y = self._downsample(np.convolve(x_pulseshaped_lowpassed, self.h_pulseshape, 'same'), samps_per_symb) * samps_per_symb
            #print('y', y.shape)

            # Demodulate
            demod_bits = hmod.demodulate(y, 'hard')

            # Convert back to integers
            res_b = self._convert_to_int_stream(demod_bits, bits_per_code)
            res_b = np.multiply(res_b, sign_mask)
            res[b] = res_b

        res = torch.from_numpy(np.array(res)).float()
        if USE_CUDA: res = res.cuda()
        ###### End of sketchy stuff #######

        # Simulated Gaussian noise.
        noise = Variable(torch.randn(*res.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        if USE_CUDA: noise = noise.cuda()
        res += noise
        
        res = self.decode(res)

        return res 

    def _downsample(self, sig, r):
        return sig[::r]

    def _upsample(self, sig, r):
        res = np.zeros(len(sig)*r, dtype=complex)
        for i in range(len(res)):
            if i % r == 0:
                res[i] = sig[int(i / r)]
        return res

    def _convert_to_bit_stream(self, x, bits_per_code, batch_size):
        res = [] 
        for e in x:
            binary = bin(e)
            b_ind = binary.index('b')
            binary = binary[b_ind+1:].zfill(bits_per_code)
            for mint in binary:
                res.append(int(mint))
        return np.array(res)

    def _convert_to_int_stream(self, x, M):
        res = []
        for i in range(0, len(x)-M+1, M):
            binary = x[i:i+M]
            res.append(int("".join(str(x) for x in binary), 2)) 
        return np.array(res)

    def _batch_modulate(self, x, modulator, batch_size):
        for b in range(batch_size):
            modulator


def accuracy(preds, labels):
    return torch.sum(torch.eq(pred, labels)).item()/(list(preds.size())[0])


if __name__ == "__main__":
    train_new_model = True 
    if train_new_model == True:
        # Data generation
        train_labels = (torch.rand(NUM_TRAINING) * (2**BLOCK_SIZE)).long()
        train_data = torch.eye(2**BLOCK_SIZE).index_select(dim=0, index=train_labels)
        test_labels = (torch.rand(1500) * (2**BLOCK_SIZE)).long()
        test_data = torch.eye(2**BLOCK_SIZE).index_select(dim=0, index=test_labels)

        # Data loading
        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 6}
        training_set = Dataset(train_data, train_labels)
        training_loader = torch.utils.data.DataLoader(training_set, **params)
        loss_fn = nn.CrossEntropyLoss()

        # Training
        snrs_db = np.linspace(-4, 9, num=14)
        snrs_db = [7]
        for snr in snrs_db:
            loss_list = []
            acc_list = []

            # The input dimension to the autoencoder should be the number of 
            # possible codewords, i.e. 2^k. This is because k represents the
            # number of bits of the block we wish to transmit, and we are 
            # representing input as a one-hot vector, so each input vector
            # will be of length 2^k.
            # The hidden dimension to the autoencoder should be n, the channel
            # use. This is because we wish the autoencoder to learn an optimal
            # coding scheme, so the middle layer should encode a vector of length
            # n.
            model = Net(2**BLOCK_SIZE, compressed_dim=CHANNEL_USE, snr=snr)
            if USE_CUDA: model = model.cuda()
            optimizer = Adam(model.parameters(), lr=0.001)

            #with imageio.get_writer('results/gifs/total_norm_16_snr_'+str(snr)+'.gif', mode='I') as writer:
            for epoch in range(NUM_EPOCHS): 
                for batch_idx, (batch, labels) in tqdm(enumerate(training_loader), total=int(training_set.__len__()/BATCH_SIZE)):
                    if USE_CUDA:
                        batch = batch.cuda()
                        labels = labels.cuda()
                    output = model(batch)
                    loss = loss_fn(output, labels)

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch_idx == int(training_set.__len__()/BATCH_SIZE)-1:
                        pred = torch.argmax(output, dim=1)
                        acc = accuracy(pred, labels)  
                        print('Epoch %2d for SNR %s: loss=%.4f, acc=%.2f' % (epoch, snr, loss.item(), acc))
                        loss_list.append(loss.item())
                        acc_list.append(acc)

                        #train_data = train_data.cuda()
                        #train_codes = model.encode(train_data)
                        #train_codes = train_codes.cpu().detach().numpy()
                        #fig = plt.figure()
                        #plt.scatter(train_codes[:, 0], train_codes[:, 1])
                        #plt.savefig('results/images/foo'+str(epoch)+'.png')
                        #fig.clf()
                        #plt.close()
                        #image = imageio.imread('results/images/foo'+str(epoch)+'.png')
                        #writer.append_data(image)


            torch.save(model.state_dict(), './models/model_with_fancy_mod'+str(snr))
            print(list(model.parameters()))

    else:
        test_labels = (torch.rand(1500) * CHANNEL_SIZE).long()
        test_data = torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)
        model = Net(CHANNEL_SIZE, compressed_dim=int(math.log2(CHANNEL_SIZE)), snr=2)
        model.load_state_dict(torch.load('./model_state'))
        model = model.cuda()
        test_data = test_data.cuda()
        test_codes = model.encode(test_data)
        test_codes = test_codes.cpu().detach().numpy()
        print(test_codes)
        plt.scatter(test_codes[:, 0], test_codes[:, 1])
        plt.savefig('foo.png')
