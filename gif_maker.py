import imageio
import os
import glob

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

files = glob.glob(FFTshift_PATH + '*.png')
files = sorted(files)
print(files)
for filename in files:
    fft_shift_images.append(imageio.imread(filename))
imageio.mimsave('./results/gifs/fft_shift.gif', fft_shift_images)

files = glob.glob(CONSTELLATION_PATH + '*.png')
files = sorted(files)
for filename in files:
    const_images.append(imageio.imread(filename))
imageio.mimsave('./results/gifs/const.gif', const_images)
