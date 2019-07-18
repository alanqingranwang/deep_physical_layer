import imageio
import matplotlib.pyplot as plt
import os
import glob
import itertools
import torch
from experimental import Net

FFT_PATH = './results/images/fft_none/'
FFTshift_PATH = './results/images/fft_shift/'
CONSTELLATION_PATH = './results/images/constellation/'
fft_images = []
fft_shift_images = []
const_images = []

files = glob.glob(FFT_PATH + '*.png')
files = sorted(files)
print(files)
for filename in files:
    fft_images.append(imageio.imread(filename))
imageio.mimsave('./results/gifs/fft_none.gif', fft_images)

# files = glob.glob(FFTshift_PATH + '*.png')
# files = sorted(files)
# print(files)
# for filename in files:
#     fft_shift_images.append(imageio.imread(filename))
# imageio.mimsave('./results/gifs/fft_shift.gif', fft_shift_images)

# files = glob.glob(CONSTELLATION_PATH + '*.png')
# files = sorted(files)
# for filename in files:
#     const_images.append(imageio.imread(filename))
# imageio.mimsave('./results/gifs/const.gif', const_images)

# block_size = 4
# channel_use = 2
# USE_CUDA = True
# snr = 8

# lst = torch.tensor(list(map(list, itertools.product([-1, 1], repeat=block_size)))).cuda()
# sample_data = torch.zeros((len(lst), block_size*2)).cuda()
# sample_data[:, :block_size] = lst
# model = Net(channel_use=channel_use, block_size=block_size, snr=-5, use_cuda=USE_CUDA, use_lpf=False, use_complex=True, dropout_rate=0)
# model.load_state_dict(torch.load('./models/4_2_' + str(snr) + '.0'))
# model.eval()
# if USE_CUDA: model = model.cuda()

# train_codes = model.encode(sample_data)
# print(train_codes.size())
# train_codes_cpu = train_codes.cpu().detach().numpy()

# for i in range(channel_use):
#     fig = plt.figure()

#     plt.scatter(train_codes_cpu[:,i], train_codes_cpu[:,i + channel_use])
#     plt.xlim([-2, 2])
#     plt.ylim([-2, 2])
#     plt.title('SNR %s, Constellation %s' % (str(snr), str(i)))
#     plt.savefig('results/images/constellation/const_%s.png' % (str(i)))
#     fig.clf()
#     plt.close()
