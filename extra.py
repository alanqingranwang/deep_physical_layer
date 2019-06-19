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

import imageio
from pulseshape_lowpass import pulseshape_lowpass
from tqdm import tqdm
import torch.nn.functional as F

# Machine learning parameters
NUM_EPOCHS = 300
BATCH_SIZE = 256

# Comms parameters
CHANNEL_USE = 7 # The parameter n
BLOCK_SIZE = 4 # The parameter k
SAMPS_PER_SYMB = 6

# Torch parameters
USE_CUDA = True

class Net(nn.Module):
    def __init__(self, in_channels, enc_compressed_dim, dec_compressed_dim, h_lowpass, snr):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.enc_compressed_dim = enc_compressed_dim
        self.dec_compressed_dim = dec_compressed_dim
        self.snr = snr
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels, enc_compressed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dec_compressed_dim, in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )
        self.h_lowpass = h_lowpass
        self.conv1 = nn.Conv1d(1, 1, len(h_lowpass), padding=len(h_lowpass)-1, bias=False)
        self.conv1.weight.data = h_lowpass.view(1, 1, -1)

        self.dropout = nn.Dropout(p=0.25)
        self.sig = nn.Sigmoid()


    def decode_signal(self, x):
        return self.decoder(x)

    def encode(self, x):
        x = self.encoder(x)
        #x = self.dropout(x)
        # Normalization so that every row of x is normalized. Forces unit amplitude...
        x = (x / x.norm(dim=-1)[:, None])

        # Normalization. Forces average power to be equal to n.
        #x = x / CHANNEL_USE
        #x = x / x.norm() * 70
        return x

    def forward(self, x):
        batch_size = len(x)
        x = self.encode(x)

        #x = pulseshape_lowpass(x, SAMPS_PER_SYMB, batch_size, self.h_lowpass, USE_CUDA)
        x = pad_within(x, stride=SAMPS_PER_SYMB)
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = torch.squeeze(x)

        # Simulated Gaussian noise.
        training_signal_noise_ratio = 10**(0.1*self.snr)

        # bit / channel_use
        communication_rate = BLOCK_SIZE / CHANNEL_USE

        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        if USE_CUDA: noise = noise.cuda()
        x += noise

        x = self.decoder(x)

        x = self.sig(x)
        return x

def pad_within(x, stride=2):
    z = torch.zeros(len(x), x.shape[1]*stride)
    index = torch.arange(0, z.shape[1], stride)
    if USE_CUDA:
        z = z.cuda()
        index = index.cuda()
    return z.index_copy_(1, index, x)

def accuracy(preds, labels, to_print=False):
    if to_print:
        print(torch.abs(preds-labels))
    # complete accuracy by sample
    acc1 = torch.sum((torch.sum(torch.abs(preds-labels), 1)==0)).item()/(list(preds.size())[0])
    # total accuracy over everything
    acc2 = 1 - torch.sum(torch.abs(preds-labels)).item() / (list(preds.size())[0]*list(preds.size())[1])
    return acc2


if __name__ == "__main__":
    train_new_model = True
    if train_new_model == True:
        # Data generation
        train_data = torch.randint(2, (10000, BLOCK_SIZE)).float()
        train_labels = train_data
        test_data = torch.randint(2, (1500, BLOCK_SIZE)).float()
        test_labels = test_data

        # Data loading
        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 6}
        training_set = Dataset(train_data, train_labels)
        training_loader = torch.utils.data.DataLoader(training_set, **params)
        loss_fn = nn.BCELoss()

        # Training
        snrs_db = np.linspace(-4, 9, num=14)
        snrs_db = [7]
        for snr in snrs_db:
            # loss_list = []
            # acc_list = []

            # The input dimension to the autoencoder is the number of bits per codeword.
            # In this version, we want to feed in symbols and not a 1-hot vector.
            # The hidden dimension to the autoencoder should be n, the channel
            # use. This is because we wish the autoencoder to learn an optimal
            # coding scheme, so the middle layer should encode a vector of length
            # n.
            #h_lowpass = np.loadtxt('h_lowpass.txt')
            h_lowpass = torch.from_numpy(signal.firwin(10, 1/2)).float()
            compressed_dim = len(h_lowpass) + CHANNEL_USE*SAMPS_PER_SYMB - 1
            model = Net(in_channels=BLOCK_SIZE, enc_compressed_dim=CHANNEL_USE, dec_compressed_dim=compressed_dim, h_lowpass=h_lowpass, snr=snr)

            model.conv1.weight.requires_grad = False

            if USE_CUDA: model = model.cuda()
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)

            for epoch in range(NUM_EPOCHS):
                for batch_idx, (batch, labels) in tqdm(enumerate(training_loader), total=int(training_set.__len__()/BATCH_SIZE)):
                    if USE_CUDA:
                        batch = batch.cuda()
                        labels = labels.cuda()
                    output = model(batch)
                    loss = loss_fn(output, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch_idx % (BATCH_SIZE-1) == 0:
                        pred = torch.round(output)
                        acc = accuracy(pred, labels)
                        print('Epoch %2d for SNR %s: loss=%.4f, acc=%.2f' % (epoch, snr, loss.item(), acc))
                        # loss_list.append(loss.item())
                        # acc_list.append(acc)

                        train_data = train_data.cuda()
                        train_codes = model.encode(train_data)
                        train_codes = train_codes.cpu().detach().numpy()
                        fig = plt.figure()
                        plt.scatter(train_codes[:, 0], train_codes[:, 1])
                        plt.title('Epoch ' + str(epoch))
                        plt.savefig('results/images/constellation/const_%3d.png' % (epoch))
                        fig.clf()
                        plt.close()

                        train_code = model.encode(train_data[0].view(1, -1))
                        train_code = train_code.cpu().detach().numpy()
                        fig = plt.figure()
                        L = 100
                        train_code_padded = np.zeros(L)
                        train_code_padded[:len(train_code[0])] = train_code[0]
                        H = np.fft.fft(train_code_padded)
                        plt.plot(np.abs(H))
                        lowpass_pad = np.zeros(L)
                        lowpass_pad[:len(h_lowpass)] = h_lowpass
                        lowpass_fft = np.fft.fft(lowpass_pad)
                        plt.plot(np.abs(lowpass_fft))
                        plt.title('Epoch ' + str(epoch))
                        plt.savefig('results/images/fft/fft_%3d.png' % (epoch))
                        fig.clf()
                        plt.close()

                # Validation
                if USE_CUDA:
                    test_data = test_data.cuda()
                    test_labels = test_labels.cuda()
                if epoch % 10 == 0:
                    val_output = model(test_data)
                    val_loss = loss_fn(val_output, test_labels)
                    val_pred = torch.round(val_output)
                    val_acc = accuracy(val_pred, test_labels, True)
                    print('Validation: Epoch %2d for SNR %s: loss=%.4f, acc=%.2f' % (epoch, snr, val_loss.item(), val_acc))

        #torch.save(model.state_dict(), './models/model_state_validation')
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
