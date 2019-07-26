# Pytorch modules
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.autograd import Variable

# User-defined modules
from model import Net
from Dataset import Dataset
from gan import Discriminator, Generator
from generate_data import generate_data

# Other modules
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pprint
import imageio
from pulseshape_lowpass import pulseshape_lowpass
from tqdm import tqdm
import itertools
import argparse
import time
import itertools
import math
import scipy.signal as signal
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser(description='Channel Model Learning')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--use_lpf', dest='use_lpf', action='store_true')
parser.add_argument('--no_lpf', dest='use_lpf', action='store_false')
parser.set_defaults(use_lpf=True)
parser.add_argument('--use_complex', dest='use_complex', action='store_true')
parser.add_argument('--use_real', dest='use_complex', action='store_false')
parser.set_defaults(use_complex=False)
parser.add_argument('--lpf_num_taps', default=100, type=int,
                    metavar='taps', help='number of lpf taps (default: 100)')
parser.add_argument('--lpf_cutoff', default=0.3, type=int,
                    metavar='cutoff', help='lpf cutoff (default: 0.3)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--dropout_rate', default=0., type=float,
                    metavar='dropout_rate', help='dropout rate')
parser.add_argument('--lpf_shift_type', default='cont', type=str,
                    metavar='shift_type', help='shift type for lpf training')
parser.add_argument('--channel_use', default=32, type=int,
                    metavar='channel_use', help='n parameter')
parser.add_argument('--block_size', default=4, type=int,
                    metavar='block_size', help='k parameter')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('-s', '--snr', default=7., type=float, metavar='float',
                    help='snr')

def main():
    args = parser.parseargs()
    pprint.pprint(vars(args))

    USE_CUDA = torch.cuda.is_available()
    train_data, train_labels, test_data, test_labels = generate_data(args.block_size, args.use_complex)

    # Data loading
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers}
    training_loader = torch.utils.data.DataLoader(Dataset(train_data, train_labels), **params)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training
    val_loss_list_normal = []
    val_acc_list_normal = []

    autoencoder = Net(channel_use=args.channel_use, block_size=args.block_size, snr=args.snr, use_cuda=USE_CUDA, use_lpf=args.use_lpf, use_complex = args.use_complex, lpf_num_taps=args.lpf_num_taps, dropout_rate=args.dropout_rate)

    generator = Generator(channel_use)
    discriminator = Discriminator(channel_use)

    if USE_CUDA:
        autoencoder = autoencoder.cuda()
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        # if args.lpf_shift_type == 'cont':
        #     cutoff = max(args.lpf_cutoff, (args.epochs-epoch-1)/args.epochs)
        # else:
        #     if epoch % (args.epochs / 10) == 0:
        #         cutoff = max(args.lpf_cutoff, (args.epochs-epoch-1)/args.epochs)


        for name, param in model.named_parameters():
            if 'conv' in name:
                param.requires_grad = False

        # filter_real = torch.randint(20, (args.lpf_num_taps,)).float()
        # filter_imag = torch.randint(20, (args.lpf_num_taps,)).float()
        # if USE_CUDA:
        #     filter_real = filter_real.cuda()
        #     filter_imag = filter_imag.cuda()
        # filter_comp = torch.stack((filter_real, filter_imag), dim=1)
        # fft = torch.ifft(filter_comp, 1)
        # # model.conv1.conv_real.weight.data = filter_real.view(1, 1, -1)
        # # model.conv1.conv_imag.weight.data = filter_imag.view(1, 1, -1)
        # model.conv1.conv_real.weight.data = fft[:, 0].view(1, 1, -1)
        # model.conv1.conv_imag.weight.data = fft[:, 1].view(1, 1, -1)

        model.train()
        for batch_idx, (batch, labels) in tqdm(enumerate(training_loader), total=int(training_set.__len__()/args.batch_size)):
            if USE_CUDA:
                batch = batch.cuda()
                labels = labels.cuda()
            output, noise = model(batch, epoch)
            loss = loss_fn(output, labels)

            train_receiver()

            train_transmitter()

            train_channel()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % (args.batch_size-1) == 0:
                pred = torch.round(torch.sigmoid(output))
                acc = model.accuracy(pred, labels)
                print('Epoch %2d for SNR %s, shift type %s: loss=%.4f, acc=%.2f' % (epoch, args.snr, args.lpf_shift_type, loss.item(), acc))
                # loss_list.append(loss.item())
                # acc_list.append(acc)


                model.eval()

                # Create gif of fft
                if args.use_complex:
                    sample = torch.zeros(args.block_size*2).float()
                else:
                    sample = torch.zeros(args.block_size).float()
                if USE_CUDA: sample = sample.cuda()
                create_fft_plots(sample, noise, model, epoch)

                # Save images of constellations
                # lst = torch.tensor(list(map(list, itertools.product([0, 1], repeat=args.block_size))))
                # if args.use_complex == True:
                #     sample_data = torch.zeros((len(lst), args.block_size*2))
                #     sample_data[:, :args.block_size] = lst
                # else:
                #     sample_data = lst
                # if USE_CUDA: sample_data = sample_data.cuda()
                # create_constellation_plots(sample_data, args.block_size, args.channel_use, model, args.snr, epoch, USE_CUDA, args.use_complex)

                # Validation
                if USE_CUDA:
                    test_data = test_data.cuda()
                    test_labels = test_labels.cuda()
                if epoch % 10 == 0:
                    val_output, _ = model(test_data, epoch)
                    val_loss = loss_fn(val_output, test_labels)
                    val_pred = torch.round(torch.sigmoid(val_output))
                    val_acc = model.accuracy(val_pred, test_labels)
                    # val_loss_list_normal.append(val_loss)
                    # val_acc_list_normal.append(val_acc)
                    print('Validation: Epoch %2d for SNR %s: loss=%.4f, acc=%.5f' % (epoch, args.snr, val_loss.item(), val_acc))
                model.train()
        if args.use_complex:
            torch.save(model.state_dict(), './models/complex_(%s,%s)_%s' % (str(args.channel_use), str(args.block_size), str(args.snr)))
        else:
            torch.save(model.state_dict(), './models/real_(%s,%s)_%s' % (str(args.channel_use), str(args.block_size), str(args.snr)))

if __name__ == "__main__":
    main()
