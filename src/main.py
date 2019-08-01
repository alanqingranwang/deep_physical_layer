'''
Alan Wang, AL29162
7/31/19
'''
# Pytorch modules
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.autograd import Variable

# User-defined modules
from Dataset import Dataset
from data_generator import generate_data
from autoencoder import Autoencoder
from rnn import RNN
from utils import create_noisytones_fft

# Other modules
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse
import time
import itertools
import scipy.signal as signal
from sklearn.manifold import TSNE
from tqdm import tqdm
import pprint
import imageio

parser = argparse.ArgumentParser(description='Channel Model Learning')

# Model type
parser.add_argument('--autoencoder', dest='use_autoencoder', action='store_true')
parser.add_argument('--rnn', dest='use_autoencoder', action='store_false')
parser.set_defaults(use_autoencoder=True)

# (n, k)
parser.add_argument('--channel_use', default=32, type=int,
                    metavar='channel_use', help='n parameter')
parser.add_argument('--block_size', default=4, type=int,
                    metavar='block_size', help='k parameter')
parser.add_argument('--snr', default=7., type=float, metavar='float',
                    help='snr')

# Channel type
parser.add_argument('--channel_type', default='awgn', type=str,
                    metavar='channel_type', help='channel to model {awgn, time-varying tones}')

# Filter specific parameters
parser.add_argument('--use_lpf', dest='use_lpf', action='store_true')
parser.add_argument('--no_lpf', dest='use_lpf', action='store_false')
parser.set_defaults(use_lpf=True)
parser.add_argument('--lpf_num_taps', default=100, type=int,
                    metavar='taps', help='number of lpf taps (default: 100)')
parser.add_argument('--lpf_cutoff', default=0.3, type=int,
                    metavar='cutoff', help='lpf cutoff (default: 0.3)')
parser.add_argument('--lpf_shift_type', default='cont', type=str,
                    metavar='shift_type', help='shift type for lpf training')

# Complex or real numbers
parser.add_argument('--use_complex', dest='use_complex', action='store_true')
parser.add_argument('--use_real', dest='use_complex', action='store_false')
parser.set_defaults(use_complex=False)

# Machine learning parameters
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

def main():
    args = parser.parse_args()
    args.device = None
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    pprint.pprint(vars(args))

    USE_CUDA = torch.cuda.is_available()
    train_data, train_labels, test_data, test_labels = generate_data(args.block_size, args.use_complex)
    if not args.use_autoencoder:
        train_data = train_data.unsqueeze(1)
        train_labels = train_labels.unsqueeze(1)
        test_data = test_data.unsqueeze(1)
        test_labels = test_labels.unsqueeze(1)

    # Data loading
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers}
    training_set = Dataset(train_data, train_labels)
    training_loader = torch.utils.data.DataLoader(training_set, **params)
    loss_fn = nn.BCEWithLogitsLoss()

    if args.use_autoencoder:
        model = Autoencoder(channel_use=args.channel_use, block_size=args.block_size, snr=args.snr, use_cuda=USE_CUDA, use_lpf=args.use_lpf, use_complex = args.use_complex, channel_type=args.channel_type)
    else:
        model = RNN(channel_use=args.channel_use, block_size=args.block_size, n_layers=1, snr=args.snr, num_taps=args.lpf_num_taps, use_cuda=USE_CUDA, channel_type=args.channel_type)

    if USE_CUDA:
        model = model.cuda()
        test_data = test_data.cuda()
        test_labels = test_labels.cuda()

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        if args.lpf_shift_type == 'cont':
            cutoff = max(args.lpf_cutoff, (args.epochs-epoch-1)/args.epochs)
        else:
            if epoch % (args.epochs / 10) == 0:
                cutoff = max(args.lpf_cutoff, (args.epochs-epoch-1)/args.epochs)


        # Filter weights (if enabled) should not change
        for name, param in model.named_parameters():
            if 'conv' in name:
                param.requires_grad = False

        model.train()
        for batch_idx, (batch, labels) in tqdm(enumerate(training_loader), total=int(training_set.__len__()/args.batch_size)):
            hidden = None
            if USE_CUDA:
                batch = batch.cuda()
                labels = labels.cuda()
            if args.use_autoencoder:
                output = model(batch)
            else:
                output, hidden = model(batch)

            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % (args.batch_size-1) == 0:
                pred = torch.round(torch.sigmoid(output))
                acc = model.accuracy(pred, labels)
                print('Epoch %2d for SNR %s, shift type %s: loss=%.4f, acc=%.2f' % (epoch, args.snr, args.lpf_shift_type, loss.item(), acc))
                # model.eval()
                # sample = torch.zeros(args.block_size).view(1, -1).cuda()
                # create_noisytones_fft(model.encode(sample), epoch, args.channel_use)
                # model.train()

        # Validation
        model.eval()
        if epoch % 10 == 0:
            if args.use_autoencoder:
                val_output = model(test_data)
            else:
                val_output, _ = model(test_data)
            val_loss = loss_fn(val_output, test_labels)
            val_pred = torch.round(torch.sigmoid(val_output))
            val_acc = model.accuracy(val_pred, test_labels)
            print('Validation: Epoch %2d for SNR %s and cutoff %s: loss=%.4f, acc=%.5f' % (epoch, args.snr, cutoff, val_loss.item(), val_acc))
        model.train()

    if args.use_complex:
        torch.save(model.state_dict(), './models/complex_(%s,%s)_%s' % (str(args.channel_use), str(args.block_size), str(args.snr)))
    else:
        torch.save(model.state_dict(), './models/real_(%s,%s)_%s' % (str(args.channel_use), str(args.block_size), str(args.snr)))

if __name__ == "__main__":
    main()
