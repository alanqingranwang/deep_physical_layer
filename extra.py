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
NUM_EPOCHS = 800
BATCH_SIZE = 256
LEARN_RATE = 0.01
DROP_RATE = 0

# Comms parameters
CHANNEL_USE = 32 # The parameter n
BLOCK_SIZE = 4 # The parameter k

# Signal processing parameters
# LPF_CUTOFF = 1/2
NUM_TAPS = 100

# Torch parameters
USE_CUDA = True



class Net(nn.Module):
    def __init__(self, in_channels, enc_compressed_dim, dec_compressed_dim, lpf_num_taps, snr):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.enc_compressed_dim = enc_compressed_dim
        self.dec_compressed_dim = dec_compressed_dim
        self.snr = snr
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.PReLU(),
            nn.BatchNorm1d(in_channels),
            nn.Dropout(p=DROP_RATE),
            nn.Linear(in_channels, enc_compressed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dec_compressed_dim, in_channels),
            nn.PReLU(),
            nn.BatchNorm1d(in_channels),
            # nn.Linear(in_channels, in_channels),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(in_channels),
            nn.Dropout(p=DROP_RATE),
            nn.Linear(in_channels, in_channels)
        )
        self.conv1 = nn.Conv1d(1, 1, lpf_num_taps, padding=lpf_num_taps-1, bias=False)
        self.sig = nn.Sigmoid()

    def decode(self, x):
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        # Normalization so that every row of x is normalized. Forces unit amplitude scaled to some factor
        x = self.in_channels * (x / x.norm(dim=-1)[:, None])

        # Normalization. Scales points such that the maximum point is at the unit circle boundary.
        # x = x / x.norm() * 2**BLOCK_SIZE * 100
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
        x = self.sig(x) # Sigmoid for BCELoss
        return x


def accuracy(preds, labels):
    # block-wise accuracy
    acc1 = torch.sum((torch.sum(torch.abs(preds-labels), 1)==0)).item()/(list(preds.size())[0])
    # bit-wise accuracy
    acc2 = 1 - torch.sum(torch.abs(preds-labels)).item() / (list(preds.size())[0]*list(preds.size())[1])
    return acc2


if __name__ == "__main__":
    # Data generation
    train_data = torch.randint(2, (10000, BLOCK_SIZE)).float()
    train_labels = train_data
    test_data = torch.randint(2, (2500, BLOCK_SIZE)).float()
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

    # Training
    snrs_db = np.linspace(-5, 10, num=20)
    # plt.figure()
    for i, snr in enumerate(snrs_db):
        if snr < 6.1:
            continue

        val_loss_list_normal = []
        val_acc_list_normal = []

        # The input dimension to the autoencoder is the number of bits per codeword.
        # In this version, we want to feed in symbols and not a 1-hot vector.
        # The hidden dimension to the autoencoder should be n, the channel
        # use. This is because we wish the autoencoder to learn an optimal
        # coding scheme, so the middle layer should encode a vector of length
        # n.
        # h_lowpass_normal = torch.tensor(np.loadtxt('sharp_h_lowpass_0.40.txt')).float().cuda()

        # The lowpass filter layer doesn't get its weights changed

        com_dim = NUM_TAPS+CHANNEL_USE-1
        model = Net(in_channels=BLOCK_SIZE, enc_compressed_dim=CHANNEL_USE, dec_compressed_dim=com_dim, lpf_num_taps=NUM_TAPS, snr=snr)

        if USE_CUDA: model = model.cuda()

        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARN_RATE)

        for epoch in range(NUM_EPOCHS):
            cutoff = max(0.3, (NUM_EPOCHS-epoch-1)/NUM_EPOCHS)
            h_lowpass = torch.from_numpy(signal.firwin(NUM_TAPS, cutoff)).float().cuda()
            model.conv1.weight.requires_grad = False
            model.conv1.weight.data = h_lowpass.view(1, 1, -1)

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

                    # Create gif of constellations
                    train_data = train_data.cuda()
                    train_codes = model.encode(train_data)
                    train_codes = train_codes.cpu().detach().numpy()
                    fig = plt.figure()
                    plt.scatter(train_codes[:, 0], train_codes[:, 1])
                    plt.title('Epoch ' + str(epoch))
                    plt.savefig('results/images/constellation/const_%3d.png' % (epoch))
                    fig.clf()
                    plt.close()

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
                    plt.savefig('results/images/fft_none/fft_%s.png' % (str(epoch).zfill(4)))
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
                        print('Validation: Epoch %2d for SNR %s and cutoff %s: loss=%.4f, acc=%.5f' % (epoch, snr, cutoff, val_loss.item(), val_acc))
                    model.train()
        torch.save(model.state_dict(), './models/model_lpf_shift_' + str(snr))
    # plt.figure()
    # plt.plot(val_loss_list_normal, label='normal')
    # plt.legend()
    # plt.savefig('./results/loss_shifting_exper.png')

    # plt.figure()
    # plt.plot(val_acc_list_normal, label='normal')
    # plt.legend()
    # plt.savefig('./results/acc_shifting_exper.png')
