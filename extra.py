import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from Dataset import Dataset
from torch.optim import Adam, SGD
import math
from torch.autograd import Variable
import scipy.signal as signal

import imageio
from pulseshape_lowpass import pulseshape_lowpass
from tqdm import tqdm
import torch.nn.functional as F
import itertools

# Machine learning parameters
NUM_EPOCHS = 1400
BATCH_SIZE = 256
LEARN_RATE = 0.01
DROP_RATE = 0.25

# Comms parameters
CHANNEL_USE = 32 # The parameter n
BLOCK_SIZE = 4 # The parameter k

# Signal processing parameters
# SAMPS_PER_SYMB = 6
# LPF_CUTOFF = 1/2

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
            nn.BatchNorm1d(in_channels),
            nn.Dropout(p=DROP_RATE),
            nn.Linear(in_channels, enc_compressed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dec_compressed_dim, in_channels),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(in_channels),
            nn.Dropout(p=DROP_RATE),
            nn.Linear(in_channels, in_channels)
        )
        self.h_lowpass = h_lowpass
        self.conv1 = nn.Conv1d(1, 1, len(h_lowpass), padding=len(h_lowpass)-1, bias=False)

        self.sig = nn.Sigmoid()


    def decode(self, x):
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        # Normalization so that every row of x is normalized. Forces unit amplitude...
        x = self.in_channels**2 * (x / x.norm(dim=-1)[:, None])

        # Normalization. Scales points such that the maximum point is at the unit circle boundary.
        # x = x / x.norm()
        # max_x = torch.max(torch.norm(x, dim=1))
        # scale_factor = 1/max_x
        # x = x*scale_factor
        return x

    def lpf(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        return x

    def awgn(self, x):
        # Simulated Gaussian noise.
        training_signal_noise_ratio = 10**(0.1*self.snr)

        # bit / channel_use
        communication_rate = BLOCK_SIZE / CHANNEL_USE

        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        if USE_CUDA: noise = noise.cuda()
        x += noise
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.lpf(x)
        x = self.awgn(x)
        x = self.decode(x)
        # Sigmoid for BCELoss
        x = self.sig(x)
        return x


def accuracy(preds, labels, to_print=False):
    if to_print:
        print(torch.abs(preds-labels))
    # complete accuracy by sample
    acc1 = torch.sum((torch.sum(torch.abs(preds-labels), 1)==0)).item()/(list(preds.size())[0])
    # total accuracy over everything
    acc2 = 1 - torch.sum(torch.abs(preds-labels)).item() / (list(preds.size())[0]*list(preds.size())[1])
    return acc2


if __name__ == "__main__":
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
    loss_fn = nn.BCEWithLogitsLoss()

    # Data for constellation generation
    sample_data = torch.tensor(list(map(list, itertools.product([0, 1], repeat=BLOCK_SIZE)))).float().cuda()
    if USE_CUDA: sample_data = sample_data.cuda()

    # Data for fft generation
    sample = torch.randint(2, (1, BLOCK_SIZE)).float().cuda()
    if USE_CUDA: sample = sample.cuda()

    h_lowpass_25 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.25.txt'), (0, 3), 'constant')).float().cuda()
    h_lowpass_30 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.30.txt'), (0, 4), 'constant')).float().cuda()
    h_lowpass_35 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.35.txt'), (0, 4), 'constant')).float().cuda()
    h_lowpass_40 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.40.txt'), (0, 0), 'constant')).float().cuda()
    h_lowpass_45 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.45.txt'), (0, 5), 'constant')).float().cuda()
    h_lowpass_50 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.50.txt'), (0, 1), 'constant')).float().cuda()
    h_lowpass_55 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.55.txt'), (0, 5), 'constant')).float().cuda()
    h_lowpass_60 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.60.txt'), (0, 5), 'constant')).float().cuda()
    h_lowpass_65 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.65.txt'), (0, 5), 'constant')).float().cuda()
    h_lowpass_70 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.70.txt'), (0, 5), 'constant')).float().cuda()
    h_lowpass_75 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.75.txt'), (0, 5), 'constant')).float().cuda()
    h_lowpass_80 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.80.txt'), (0, 5), 'constant')).float().cuda()
    h_lowpass_85 = torch.tensor(np.pad(np.loadtxt('sharp_h_lowpass_0.85.txt'), (0, 3), 'constant')).float().cuda()

    # Training
    snrs_db = np.linspace(-4, 9, num=14)
    plt.figure()
    for i, snr in enumerate(snrs_db):
        val_loss_list_normal = []
        val_acc_list_normal = []

        # The input dimension to the autoencoder is the number of bits per codeword.
        # In this version, we want to feed in symbols and not a 1-hot vector.
        # The hidden dimension to the autoencoder should be n, the channel
        # use. This is because we wish the autoencoder to learn an optimal
        # coding scheme, so the middle layer should encode a vector of length
        # n.
        # h_lowpass = torch.from_numpy(signal.firwin(100, LPF_CUTOFF)).float()
        h_lowpass_normal = torch.tensor(np.loadtxt('sharp_h_lowpass_0.40.txt')).float().cuda()

        model = Net(in_channels=BLOCK_SIZE, enc_compressed_dim=CHANNEL_USE, dec_compressed_dim=len(h_lowpass_normal) + CHANNEL_USE - 1, h_lowpass=h_lowpass_normal, snr=snr)

        # The lowpass filter layer doesn't get its weights changed
        model.conv1.weight.data = h_lowpass_normal.view(1, 1, -1)
        model.conv1.weight.requires_grad = False

        if USE_CUDA: model = model.cuda()
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARN_RATE)

        for epoch in range(NUM_EPOCHS):
            if epoch < 100:
                model.conv1.weight.data = h_lowpass_85.view(1, 1, -1)
            elif epoch < 200:
                model.conv1.weight.data = h_lowpass_80.view(1, 1, -1)
            elif epoch < 300:
                model.conv1.weight.data = h_lowpass_75.view(1, 1, -1)
            elif epoch < 400:
                model.conv1.weight.data = h_lowpass_70.view(1, 1, -1)
            elif epoch < 500:
                model.conv1.weight.data = h_lowpass_65.view(1, 1, -1)
            elif epoch < 600:
                model.conv1.weight.data = h_lowpass_60.view(1, 1, -1)
            elif epoch < 700:
                model.conv1.weight.data = h_lowpass_55.view(1, 1, -1)
            elif epoch < 800:
                model.conv1.weight.data = h_lowpass_50.view(1, 1, -1)
            elif epoch < 900:
                model.conv1.weight.data = h_lowpass_45.view(1, 1, -1)
            elif epoch < 1000:
                model.conv1.weight.data = h_lowpass_40.view(1, 1, -1)
            elif epoch < 1100:
                model.conv1.weight.data = h_lowpass_35.view(1, 1, -1)
            elif epoch < 1200:
                model.conv1.weight.data = h_lowpass_30.view(1, 1, -1)
            else:
                model.conv1.weight.data = h_lowpass_25.view(1, 1, -1)

            model.train()
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

                    model.eval()

                    # # Create gif of constellations
                    # train_data = train_data.cuda()
                    # train_codes = model.encode(train_data)
                    # train_codes = train_codes.cpu().detach().numpy()
                    # fig = plt.figure()
                    # plt.scatter(train_codes[:, 0], train_codes[:, 1])
                    # plt.title('Epoch ' + str(epoch))
                    # plt.savefig('results/images/constellation/const_%3d.png' % (epoch))
                    # fig.clf()
                    # plt.close()

                    # Create gif of fft
                    sample = torch.tensor([1., 0., 1., 0.]).cuda()
                    train_code = model.encode(sample.view(1, -1))
                    fig = plt.figure()
                    L = 100
                    train_code_pad = torch.zeros(L)
                    train_code_pad[:len(train_code[0])] = train_code[0]
                    H = torch.rfft(train_code_pad, 1, normalized=True).cpu().detach().numpy()
                    plt.plot([np.sqrt(H[i, 0]**2 + H[i, 1]**2) for i in range(len(H))])

                    lowpass_coeff = model.conv1.weight.data.view(-1)
                    lowpass_pad = torch.zeros(L)
                    lowpass_pad[:len(lowpass_coeff)] = lowpass_coeff
                    lowpass_fft = torch.rfft(lowpass_pad, 1, normalized=False).cpu().detach().numpy()
                    plt.plot([np.sqrt(lowpass_fft[i, 0]**2 + lowpass_fft[i, 1]**2) for i in range(len(lowpass_fft))])
                    plt.title('Epoch ' + str(epoch))
                    plt.savefig('results/images/fft_none/fft_%3d.png' % (epoch))
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
                        val_acc = accuracy(val_pred, test_labels)
                        val_loss_list_normal.append(val_loss)
                        val_acc_list_normal.append(val_acc)
                        print('Validation: Epoch %2d for SNR %s: loss=%.4f, acc=%.2f' % (epoch, snr, val_loss.item(), val_acc))
                    model.train()
    plt.figure()
    plt.plot(val_loss_list_normal, label='normal')
    plt.plot(val_loss_list_shift, label='shift')
    plt.legend()
    plt.savefig('./results/loss_shifting_exper.png')

    plt.figure()
    plt.plot(val_acc_list_normal, label='normal')
    plt.plot(val_acc_list_shift, label='shift')
    plt.legend()
    plt.savefig('./results/acc_shifting_exper.png')
    # torch.save(model.state_dict(), './models/model_state_lpf_' + str(snr))
