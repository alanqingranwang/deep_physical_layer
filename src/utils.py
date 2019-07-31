'''
Alan Wang, AL29162
7/31/19

Various functions for visualizing codes, like producing
FFTs and constellation plots. Not guaranteed to be stable.
'''
import matplotlib.pyplot as plt

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
    plt.savefig('results/images/fft_none/fft_%s.png' % (str(epoch).zfill(4)))
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

    # # Add noise on top
    # snr_lin = 10**(0.1*snr)
    # rate = block_size / channel_use
    # if use_complex:
    #     noisiness = np.zeros((100, channel_use*2))
    # else:
    #     noisiness = np.zeros((100, channel_use))
    # for i in range(10):
    #     noisy_codes = train_codes.clone()
    #     noise = torch.randn(*noisy_codes.size()) * np.sqrt(1/(2 * rate * snr_lin))
    #     if use_cuda: noise = noise.cuda()
    #     noisy_codes += noise
    #     noisy_codes_cpu = noisy_codes.cpu().detach().numpy()
    #     noisiness[i] = noisy_codes_cpu

    # colors = itertools.cycle(['g', 'b', 'k', 'c'])
    # for i in range(channel_use):
    #     if use_complex:
    #         plt.scatter(noisiness[:,i], noisiness[:,i+channel_use], s=10, color=next(colors))
    #     else:
    #         plt.scatter(noisiness[:,i].real, noisiness[:,i].imag, s=10, color=next(colors))

    # if use_complex:
    #     plt.scatter(train_codes_cpu[:,0], train_codes_cpu[:,0 + channel_use])
    #     # for i in range(train_codes_cpu.shape[1] // 2):
    #     #     plt.scatter(train_codes_cpu[:,i], train_codes_cpu[:,i + channel_use])
    # else:
    #     plt.scatter(train_codes_cpu.real, train_codes_cpu.imag, c='r')
    plt.xlim([-500, 500])
    plt.ylim([-500, 500])
    plt.savefig('results/images/constellation/const_%s_%s.png' % (str(snr).zfill(2), str(epoch).zfill(4)))
    fig.clf()
    plt.close()
