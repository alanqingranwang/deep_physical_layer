def calc_energy(self, x):
    x_comp = x.view(-1, 2, x.shape[1] // 2)
    x_abs = torch.norm(x_comp, dim=1)
    x_sq = torch.mul(x_abs, x_abs)
    e = torch.sum(x_sq, dim=1)
    print(e)

def create_fft_plots(sample, noise, model, epoch):
    train_code = model.encode(sample.view(1, -1))
    train_code = train_code[0]
    boundary = len(train_code)//2
    code_real_pad = torch.zeros(200)
    code_imag_pad = torch.zeros(200)
    code_real_pad[:boundary] = train_code[:boundary]
    code_imag_pad[:boundary] = train_code[boundary:]
    train_code_complex = torch.stack((code_real_pad, code_imag_pad), dim=1)
    H = torch.fft(train_code_complex, 1, normalized=False).cpu().detach().numpy()

    fig = plt.figure()
    plt.plot([np.sqrt(H[i, 0]**2 + H[i, 1]**2) for i in range(len(H))])

    # filter_real = model.conv1.conv_real.weight.data.view(-1)
    # filter_imag = model.conv1.conv_imag.weight.data.view(-1)
    # # lowpass_pad = torch.zeros(L)
    # # lowpass_pad[:len(lowpass_coeff)] = lowpass_coeff
    # filter_complex = torch.stack((filter_real, filter_imag), dim=1)
    # print(filter_complex.shape)
    # lowpass_fft = torch.fft(filter_complex, 1, normalized=False).cpu().detach().numpy()

    noise_boundary = len(train_code)//2
    noise_real_pad = torch.zeros(200)
    noise_imag_pad = torch.zeros(200)
    noise_real_pad[:noise_boundary] = noise[:noise_boundary]
    noise_imag_pad[:noise_boundary] = noise[noise_boundary:]
    noise_complex = torch.stack((noise_real_pad, noise_imag_pad), dim=1)
    N = torch.fft(noise_complex, 1, normalized=True).cpu().detach().numpy()
    plt.plot([np.sqrt(N[i, 0]**2 + N[i, 1]**2) for i in range(len(N))])


    legend_strings = []
    legend_strings.append('Encoded Signal')
    legend_strings.append('Convolved Noise')
    plt.legend(legend_strings, loc = 'upper right')
    plt.title('Epoch %s' % str(epoch))
    plt.xlabel('Frequency Spectrum')
    plt.ylabel('Magnitude')
    plt.savefig('./results/images/awgn_pics/pic%s' % str(epoch).zfill(3))
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
    plt.savefig('results/images/constellation/const_%s_%s.png' % (str(snr).zfill(2), str(epoch).zfill(4)))
    fig.clf()
    plt.close()
