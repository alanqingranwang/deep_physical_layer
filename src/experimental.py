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
from channel import channel

# Other modules
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pprint
import imageio
from tqdm import tqdm
import itertools
import argparse
import math
from scipy.signal import firwin
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

def train_receiver(batch, labels, test_data, test_labels, batch_idx, autoencoder, loss_fn, optimizer, epoch, snr, cuda):
    # Freeze weights of encoder
    # because decoder is being trained
    for name, param in autoencoder.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
        if 'decoder' in name:
            param.requires_grad = True

    output = autoencoder(batch, epoch, None, snr)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch_idx == 0:
        pred = torch.round(torch.sigmoid(output))
        acc = autoencoder.accuracy(pred, labels)
        print('Epoch %2d for SNR %s: loss=%.4f, acc=%.2f' % (epoch, snr, loss.item(), acc))

        autoencoder.eval()

        # Validation
        if cuda:
            test_data = test_data.cuda()
            test_labels = test_labels.cuda()
        if epoch % 10 == 0:
            val_output = autoencoder(test_data, epoch, None, snr)
            val_loss = loss_fn(val_output, test_labels)
            val_pred = torch.round(torch.sigmoid(val_output))
            val_acc = autoencoder.accuracy(val_pred, test_labels)
            # val_loss_list_normal.append(val_loss)
            # val_acc_list_normal.append(val_acc)
            print('Validation: Epoch %2d for SNR %s: loss=%.4f, acc=%.5f' % (epoch, snr, val_loss.item(), val_acc))
        autoencoder.train()
    return autoencoder, optimizer

def train_transmitter(batch, labels, test_data, test_labels, batch_idx, autoencoder, channel_model, loss_fn, optimizer, epoch, snr, cuda):
    # Encoder is being trained...
    for name, param in autoencoder.named_parameters():
        if 'decoder' in name:
            param.requires_grad = False 
        if 'encoder' in name:
            param.requires_grad = True
    # But channel generator model is not
    for param in channel_model.parameters():
        param.requires_grad = False

    output = autoencoder(batch, epoch, channel_model, snr)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch_idx == 0:
        pred = torch.round(torch.sigmoid(output))
        acc = autoencoder.accuracy(pred, labels)
        print('Epoch %2d for SNR %s: loss=%.4f, acc=%.4f' % (epoch, snr, loss.item(), acc))
        # loss_list.append(loss.item())
        # acc_list.append(acc)

        autoencoder.eval()

        # Create gif of fft
        # if args.use_complex:
        #     sample = torch.zeros(args.block_size*2).float()
        # else:
        #     sample = torch.zeros(args.block_size).float()
        # if USE_CUDA: sample = sample.cuda()
        # create_fft_plots(sample, noise, model, epoch)

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
        if cuda:
            test_data = test_data.cuda()
            test_labels = test_labels.cuda()
        if epoch % 10 == 0:
            val_output = autoencoder(test_data, epoch, channel_model, snr)
            val_loss = loss_fn(val_output, test_labels)
            val_pred = torch.round(torch.sigmoid(val_output))
            val_acc = autoencoder.accuracy(val_pred, test_labels)
            # val_loss_list_normal.append(val_loss)
            # val_acc_list_normal.append(val_acc)
            print('Validation: Epoch %2d for SNR %s: loss=%.4f, acc=%.5f' % (epoch, snr, val_loss.item(), val_acc))
        autoencoder.train()
    return autoencoder, optimizer

def train_channel(batch, batch_idx, epoch, generator, discriminator, channel_use, adversarial_loss, optimizer_G, optimizer_D, n_epochs, snr, cuda):
    for param in generator.parameters():
        param.requires_grad = True

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Adversarial ground truths
    valid = Variable(Tensor(batch.size(0), 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(batch.size(0), 1).fill_(0.0), requires_grad=False)

    # Configure input
    real_codes = channel(batch, channel_use, snr)

    # -----------------
    #  Train Generator
    # -----------------
    optimizer_G.zero_grad()
    # Generate a batch of images
    gen_codes = generator(batch)

    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_codes), valid)
    g_loss.backward()
    optimizer_G.step()
    # ---------------------
    #  Train Discriminator
    # ---------------------
    optimizer_D.zero_grad()
    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_codes), valid)
    fake_loss = adversarial_loss(discriminator(gen_codes.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2
    d_acc = discriminator.accuracy(real_codes, gen_codes)

    d_loss.backward()
    optimizer_D.step()

    if batch_idx == 0:
        print("[Epoch %d/%d] [D loss: %f] [G loss: %f] [D acc: %f]"
                % (epoch, n_epochs, d_loss.item(), g_loss.item(), d_acc))

    return generator, discriminator, optimizer_G, optimizer_D


def main():
    args = parser.parse_args()
    pprint.pprint(vars(args))

    cuda = torch.cuda.is_available()
    train_data, train_labels, test_data, test_labels = generate_data(args.block_size, args.use_complex)

    # Data loading
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers}
    dataset = Dataset(train_data, train_labels)
    training_loader = torch.utils.data.DataLoader(dataset, **params)
    autoencoder_loss = nn.BCEWithLogitsLoss()
    gan_loss = torch.nn.BCELoss()

    # Training
    # val_loss_list_normal = []
    # val_acc_list_normal = []

    autoencoder = Net(channel_use=args.channel_use, block_size=args.block_size, snr=args.snr, use_cuda=cuda, use_lpf=args.use_lpf, use_complex = args.use_complex, lpf_num_taps=args.lpf_num_taps, dropout_rate=args.dropout_rate)

    generator = Generator(args.channel_use)
    discriminator = Discriminator(args.channel_use)

    if cuda:
        autoencoder = autoencoder.cuda()
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    optimizer = Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=args.lr)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        autoencoder.train()
        generator.train()
        discriminator.train()
        for batch_idx, (batch, labels) in tqdm(enumerate(training_loader), total=int(dataset.__len__()/args.batch_size)):
            if cuda:
                batch = batch.cuda()
                labels = labels.cuda()

            autoencoder, optimizer = train_receiver(batch, labels, test_data, test_labels, batch_idx, autoencoder, autoencoder_loss, optimizer, epoch, args.snr, cuda)

            autoencoder, optimizer = train_transmitter(batch, labels, test_data, test_labels, batch_idx, autoencoder, generator, autoencoder_loss, optimizer, epoch, args.snr, cuda)

            generator, discriminator, optimizer_G, optimizer_D = train_channel(batch, batch_idx, epoch, generator, discriminator, args.channel_use, gan_loss, optimizer_G, optimizer_D, args.epochs, args.snr, cuda)

if __name__ == "__main__":
    main()
