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

from model import Net
import pprint
import imageio
from pulseshape_lowpass import pulseshape_lowpass
from tqdm import tqdm
import torch.nn.functional as F
import itertools
import argparse
import time

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

def get_args():
    return parser.parse_args()

def generate_data(block_size, use_complex):
    if use_complex:
        train_data = torch.randint(2, (10000, block_size*2)).float()
        test_data = torch.randint(2, (2500, block_size*2)).float()
    else:
        train_data = torch.randint(2, (10000, block_size)).float()
        test_data = torch.randint(2, (2500, block_size)).float()

    train_labels = train_data
    test_labels = test_data
    return train_data, train_labels, test_data, test_labels

def create_fft_plots(sample, model, epoch):
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

def create_constellation_plots(sample_data, model, snr, epoch):
    # Create gif of constellations
    sample_data = torch.unsqueeze(torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.]), 0).cuda()
    train_codes = model.encode(sample_data)
    train_codes = train_codes.cpu().detach().numpy()
    fig = plt.figure()
    plt.scatter(train_codes[:,:32], train_codes[:,32:])
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.title('SNR %s, Epoch %s' % (str(snr), str(epoch)))
    plt.savefig('results/images/constellation/const_%s_%s.png' % (str(snr).zfill(2), str(epoch).zfill(4)))
    fig.clf()
    plt.close()

def main():
    args = get_args()
    pprint.pprint(vars(args))

    USE_CUDA = torch.cuda.is_available()
    train_data, train_labels, test_data, test_labels = generate_data(args.block_size, args.use_complex)

    # Data loading
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers}
    training_set = Dataset(train_data, train_labels)
    training_loader = torch.utils.data.DataLoader(training_set, **params)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training
    snrs_db = np.linspace(-5, 10, num=10)
    for i, snr in enumerate(snrs_db):
        val_loss_list_normal = []
        val_acc_list_normal = []

        model = Net(channel_use=args.channel_use, block_size=args.block_size, snr=snr, use_cuda=USE_CUDA, use_lpf=args.use_lpf, use_complex = args.use_complex, dropout_rate=args.dropout_rate)

        if USE_CUDA: model = model.cuda()

        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        for epoch in range(args.epochs):
            if args.lpf_shift_type == 'cont':
                cutoff = max(args.lpf_cutoff, (args.epochs-epoch-1)/args.epochs)
            else:
                if epoch % (args.epochs / 10) == 0:
                    cutoff = max(args.lpf_cutoff, (args.epochs-epoch-1)/args.epochs)

            h_lowpass = torch.from_numpy(signal.firwin(args.lpf_num_taps, cutoff)).float()
            if USE_CUDA: h_lowpass = h_lowpass.cuda()
            model.conv1.weight.data = h_lowpass.view(1, 1, -1)
            model.conv1.weight.requires_grad = False # The lowpass filter layer doesn't get its weights changed

            model.train()
            for batch_idx, (batch, labels) in tqdm(enumerate(training_loader), total=int(training_set.__len__()/args.batch_size)):
                if USE_CUDA:
                    batch = batch.cuda()
                    labels = labels.cuda()
                output = model(batch)
                loss = loss_fn(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % (args.batch_size-1) == 0:
                    pred = torch.round(output)
                    acc = model.accuracy(pred, labels)
                    print('Epoch %2d for SNR %s, shift type %s: loss=%.4f, acc=%.2f' % (epoch, snr, args.lpf_shift_type, loss.item(), acc))
                    # loss_list.append(loss.item())
                    # acc_list.append(acc)

                    model.eval()

                    # Create gif of fft
                    # sample = torch.randint(2, (1, args.block_size)).float()
                    # if USE_CUDA: sample = sample.cuda()
                    # create_fft_plots(sample, model, epoch)

                    # Create gif of constellations
                    # sample_data = torch.tensor(list(map(list, itertools.product([0, 1], repeat=args.block_size)))).float()
                    # if USE_CUDA: sample_data = sample_data.cuda()
                    # create_constellation_plots(sample_data, model, snr, epoch)

                    # Validation
                    if USE_CUDA:
                        test_data = test_data.cuda()
                        test_labels = test_labels.cuda()
                    if epoch % 10 == 0:
                        val_output = model(test_data)
                        val_loss = loss_fn(val_output, test_labels)
                        val_pred = torch.round(val_output)
                        val_acc = model.accuracy(val_pred, test_labels)
                        # val_loss_list_normal.append(val_loss)
                        # val_acc_list_normal.append(val_acc)
                        print('Validation: Epoch %2d for SNR %s and cutoff %s: loss=%.4f, acc=%.5f' % (epoch, snr, cutoff, val_loss.item(), val_acc))
                    model.train()
            torch.save(model.state_dict(), './models/with_batch_norm_%s' % str(snr))

if __name__ == "__main__":
    main()
