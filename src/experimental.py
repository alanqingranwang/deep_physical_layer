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


def print_decoder_weights(pre,autoencoder):
    for name, param in autoencoder.named_parameters():
        if name == 'decoder.0.weight':
            print(pre, param.data)
            print(param.requires_grad)

def train_receiver(batch, labels, batch_idx, autoencoder, loss_fn, optimizer, epoch, snr, cuda):
    if batch_idx == 0 and epoch != 0:
        print(epoch, 'THE BEGINNING')
        saved = np.loadtxt('./saved_params.p')
        for name, param in autoencoder.named_parameters():
            if name == 'decoder.0.weight':
                print(saved == param.data.cpu().detach().numpy())
    # Freeze weights of encoder
    # because decoder is being trained
    # for name, param in autoencoder.named_parameters():
    #     if 'encoder' in name:
    #         param.requires_grad = False
    #     if 'decoder' in name:
    #         param.requires_grad = True
    ct = 0
    for child in autoencoder.children():
        if ct == 0:
            for name, param in child.named_parameters():
                # print(name)
                param.requires_grad = False
        if ct == 1:
            for param in child.parameters():
                param.requires_grad = True
        ct += 1

    # print_decoder_weights('BEGINNING OF TRAIN_RECEVIER', autoencoder)
    output = autoencoder(batch, epoch, None, snr, cuda)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred = torch.round(torch.sigmoid(output))
    acc = autoencoder.accuracy(pred, labels)
    # print_decoder_weights('AFTER OPTIMIZERSTEP IN RECEIVER', autoencoder)
    if batch_idx == 0:
        pred = torch.round(torch.sigmoid(output))
        acc = autoencoder.accuracy(pred, labels)
        print('RECEIVER, Epoch %2d for SNR %s: loss=%.4f, acc=%.2f' % (epoch, snr, loss.item(), acc))

        autoencoder.train()

    if batch_idx == 39:
        print(epoch, "THE END")
        for name, param in autoencoder.named_parameters():
            if name == 'decoder.0.weight':
                np.savetxt('./saved_params.p', param.data)
    # print_decoder_weights('RIGHT BEFORE EXITING RECEIVER', autoencoder)
    return autoencoder, optimizer

def train_transmitter(batch, labels, batch_idx, autoencoder, channel_model, loss_fn, optimizer, epoch, snr, cuda):
    # Freeze decoder weights
    ct = 0
    for child in autoencoder.children():
        if ct == 0:
            for name, param in child.named_parameters():
                # print(name)
                param.requires_grad = True
        if ct == 1:
            for param in child.parameters():
                param.requires_grad = False
        ct += 1

    for child in channel_model.children():
        for param in child.parameters():
            param.requires_grad = False
    # for param in channel_model.parameters():
    #     param.requires_grad = False

    # print_decoder_weights('BEGINNING OF TRANSMITTER', autoencoder)
    output = autoencoder(batch, epoch, channel_model, snr, cuda)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print_decoder_weights('AFTER OPTIMIZERSTEP IN TRANSMITTER', autoencoder)
    if batch_idx == 0:
        pred = torch.round(torch.sigmoid(output))
        acc = autoencoder.accuracy(pred, labels)
        print('TRANSMITTER, Epoch %2d for SNR %s: loss=%.4f, acc=%.4f' % (epoch, snr, loss.item(), acc))
        # loss_list.append(loss.item())
        # acc_list.append(acc)

        autoencoder.train()
    # print_decoder_weights('RIGHT BEFORE EXITING TRANSMITTER', autoencoder)
    return autoencoder, optimizer

def train_channel(batch, batch_idx, epoch, generator, discriminator, channel_use, adversarial_loss, optimizer_G, optimizer_D, n_epochs, snr, cuda):
    for param in generator.parameters():
        param.requires_grad = True

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Adversarial ground truths
    valid = Variable(Tensor(batch.size(0), 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(batch.size(0), 1).fill_(0.0), requires_grad=False)

    # Configure input
    real_codes = channel(batch, channel_use, snr, cuda)

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
    d_acc = discriminator.accuracy(real_codes, gen_codes, cuda)

    d_loss.backward()
    optimizer_D.step()

    if batch_idx == 0:
        print("[Epoch %d/%d] [D loss: %f] [G loss: %f] [D acc: %f]"
                % (epoch, n_epochs, d_loss.item(), g_loss.item(), d_acc))

    return generator, discriminator, optimizer_G, optimizer_D

def train_with_learned_generator(batch, labels, batch_idx, autoencoder, channel_model, loss_fn, optimizer, epoch, snr, cuda):
    output = autoencoder(batch, epoch, channel_model, snr, cuda)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred = torch.round(torch.sigmoid(output))
    acc = autoencoder.accuracy(pred, labels)
    if batch_idx == 0:
        print('LEARNED GENERATOR, Epoch %2d for SNR %s: loss=%.4f, acc=%.4f' % (epoch, snr, loss.item(), acc))
        autoencoder.train()
    # print_decoder_weights('RIGHT BEFORE EXITING TRANSMITTER', autoencoder)
    return autoencoder, optimizer, acc

def train_with_channel(batch, labels, batch_idx, autoencoder, loss_fn, optimizer, epoch, snr, cuda):
    output = autoencoder(batch, epoch, None, snr, cuda)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred = torch.round(torch.sigmoid(output))
    acc = autoencoder.accuracy(pred, labels)
    if batch_idx == 0:
        print('ACTUAL CHANNEL, Epoch %2d for SNR %s: loss=%.4f, acc=%.4f' % (epoch, snr, loss.item(), acc))

    return autoencoder, optimizer, acc

def validation(autoencoder, test_data, test_labels, epoch, snr, cuda):
    autoencoder.eval()
    out = autoencoder(test_data, epoch, None, snr, cuda)
    val_pred = torch.round(torch.sigmoid(out))
    val_acc = autoencoder.accuracy(val_pred, test_labels)
    print('     Validation: Epoch %2d for SNR %s: acc=%.5f' % (epoch, snr, val_acc))
    autoencoder.train()
    return val_acc

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

    # autoencoder = Net(channel_use=args.channel_use, block_size=args.block_size, snr=args.snr, use_cuda=cuda, use_lpf=args.use_lpf, use_complex = args.use_complex, lpf_num_taps=args.lpf_num_taps, dropout_rate=args.dropout_rate)

    # generator = Generator(args.channel_use)
    # discriminator = Discriminator(args.channel_use)

    if cuda:
        autoencoder = autoencoder.cuda()
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        test_data = test_data.cuda()
        test_labels = test_labels.cuda()



    autoencoder = Net(channel_use=args.channel_use, block_size=args.block_size, snr=args.snr, use_cuda=cuda, use_lpf=args.use_lpf, use_complex = args.use_complex, lpf_num_taps=args.lpf_num_taps, dropout_rate=args.dropout_rate)
    autoencoder1 = Net(channel_use=args.channel_use, block_size=args.block_size, snr=args.snr, use_cuda=cuda, use_lpf=args.use_lpf, use_complex = args.use_complex, lpf_num_taps=args.lpf_num_taps, dropout_rate=args.dropout_rate)

    generator = Generator(args.channel_use)
    generator.load_state_dict(torch.load('../models/generator'))
    for param in generator.parameters():
        param.requires_grad = False

    # discriminator = Discriminator(args.channel_use)

    # optimizer = Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=args.lr)
    optimizer = Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=args.lr)
    optimizer1 = Adam(filter(lambda p: p.requires_grad, autoencoder1.parameters()), lr=args.lr)
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    learned_acc_list = []
    channel_acc_list = []
    for epoch in range(args.epochs):
        autoencoder.train()
        # generator.train()
        # discriminator.train()
        for batch_idx, (batch, labels) in tqdm(enumerate(training_loader), total=int(dataset.__len__()/args.batch_size)):
            if cuda:
                batch = batch.cuda()
                labels = labels.cuda()

            # autoencoder, optimizer = train_receiver(batch, labels, batch_idx, autoencoder, autoencoder_loss, optimizer, epoch, args.snr, cuda)

            # autoencoder, optimizer = train_transmitter(batch, labels, batch_idx, autoencoder, generator, autoencoder_loss, optimizer, epoch, args.snr, cuda)

            autoencoder, optimizer, learned_acc = train_with_learned_generator(batch, labels, batch_idx, autoencoder, generator, autoencoder_loss, optimizer, epoch, args.snr, cuda)
            autoencoder1, optimizer1, channel_acc = train_with_channel(batch, labels, batch_idx, autoencoder1, autoencoder_loss, optimizer1, epoch, args.snr, cuda)
            learned_acc_list.append(learned_acc)
            channel_acc_list.append(channel_acc)
            # generator, discriminator, optimizer_G, optimizer_D = train_channel(batch, batch_idx, epoch, generator, discriminator, args.channel_use, gan_loss, optimizer_G, optimizer_D, args.epochs, args.snr, cuda)


            # if epoch % 10 == 0 and batch_idx == 0:
            #     acc = validation(autoencoder, test_data, test_labels, epoch, args.snr, cuda)
            #     acc_list.append(acc)

    plt.plot(learned_acc_list)
    plt.plot(channel_acc_list)
    plt.savefig('./compare_channels.png')

if __name__ == "__main__":
    main()
