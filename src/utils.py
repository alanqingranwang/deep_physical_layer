'''
Alan Wang, AL29162
7/31/19

Various functions for visualizing codes, like producing
FFTs and constellation plots. Not guaranteed to be stable.
'''
import matplotlib.pyplot as plt
import numpy as np
import torch

def create_fft_plots(sample, model, epoch):
    train_code = model.encode(sample.view(1, -1))
    train_code = train_code[0]
    fig = plt.figure()
    train_code_pad = torch.zeros(100).cuda()
    train_code_pad[:len(train_code)] = train_code
    train_code_complex = torch.stack((train_code_pad, torch.zeros(*train_code_pad.size()).cuda()), dim=1).cuda()
    H = torch.fft(train_code_complex, 1, normalized=True).cpu().detach().numpy()
    plt.plot([np.sqrt(H[i, 0]**2 + H[i, 1]**2) for i in range(len(H))])

    filter_real = model.conv1.conv_real.weight.data.view(-1)
    filter_imag = model.conv1.conv_imag.weight.data.view(-1)
    # lowpass_pad = torch.zeros(L)
    # lowpass_pad[:len(lowpass_coeff)] = lowpass_coeff
    filter_complex = torch.stack((filter_real, filter_imag), dim=1)
    print(filter_complex.shape)
    lowpass_fft = torch.fft(filter_complex, 1, normalized=False).cpu().detach().numpy()
    plt.plot([np.sqrt(lowpass_fft[i, 0]**2 + lowpass_fft[i, 1]**2) for i in range(len(lowpass_fft))])
    plt.title('Epoch ' + str(epoch))
    plt.savefig('../results/images/fft_none/fft_%s.png' % (str(epoch).zfill(4)))
    fig.clf()
    plt.close()

def create_constellation_plots(sample_data, block_size, channel_use, model, snr, epoch, use_cuda, use_complex):
    # Create gif of constellations
    train_codes = model.encode(sample_data)
    train_codes_cpu = train_codes.cpu().detach().numpy()
    embed = TSNE().fit_transform(train_codes_cpu)

    fig = plt.figure()
    plt.scatter(embed[:,0], embed[:,1])
    plt.title('t-SNE Embedding of (%s, %s) Encoding, Epoch %s' % (channel_use, block_size, str(epoch)))

    plt.xlim([-500, 500])
    plt.ylim([-500, 500])
    plt.savefig('../results/images/constellation/const_%s_%s.png' % (str(snr).zfill(2), str(epoch).zfill(4)))
    fig.clf()
    plt.close()

def create_noisytones_fft(hidden, epoch, hidden_dim, discrete_jumps=5):
    hi = hidden.cpu().detach().numpy()
    hidden = hi[0]

    L = np.linspace(0, np.pi, num=discrete_jumps)
    sig = 0
    for omega in L:
        t = np.linspace(0, 63, num=hidden_dim)
        noise_tone = torch.tensor(np.sin(omega * t)).float()
        sig += noise_tone
        noise = sig
        noise = noise.detach().numpy()
        hidden_zp = np.zeros(200)
        noise_zp = np.zeros(200)
        hidden_zp[:len(hidden)] = hidden
        noise_zp[:len(noise)] = noise

        H = np.fft.fft(hidden_zp) / len(hidden_zp)
        N = np.fft.fft(noise_zp) / len(noise_zp)
        fig = plt.figure()
        plt.plot([np.abs(H[i]) for i in range(len(H))])
        plt.plot([np.abs(N[i]) for i in range(len(N))])
        plt.title('RNN (32, 4), Epoch %s' % str(epoch))
        legend_strings = []
        legend_strings.append('Encoded Signal')
        legend_strings.append('Time-Varying Noise Tones, Superimposed')
        plt.legend(legend_strings, loc = 'upper right')
        plt.ylim([0, 0.2]) #
        plt.savefig('../results/images/rnn_pics/pics' + str(epoch).zfill(4))
        plt.close()
        fig.clf()
