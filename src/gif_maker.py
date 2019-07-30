import imageio
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import glob
import numpy as np

ALLY_PATH = '../results/images/ally_pics/'
ALLY_PATH1 = '../results/images/ally_pics1/'
ally_images = []
ally_images1 = []

loss = np.loadtxt('./loss.dat', delimiter = ',')
rate = np.loadtxt('./rate.dat', delimiter = ',')
print(rate.shape)
print(loss.shape)
for i in range(len(loss)):
    fig = plt.figure()
    plt.plot(loss[:i, 0], loss[:i, 1])
    plt.title('Loss')
    plt.ylabel('Loss Fraction')
    plt.xlabel('Time (sec)')
    plt.ylim([0, 1.1])
    plt.xlim([0, 60])
    plt.savefig(ALLY_PATH + str(i).zfill(3))
    fig.clf()
    plt.close()

files = glob.glob(ALLY_PATH + '*.png')
files = sorted(files)
print(files)
for filename in files:
    ally_images.append(imageio.imread(filename))
imageio.mimsave('./ally_loss.gif', ally_images)

for i in range(len(rate)):
    fig = plt.figure()
    plt.plot(rate[:i, 0], rate[:i, 1])
    plt.title('Rate')
    plt.ylabel('Rate (kbps)')
    plt.xlabel('Time (sec)')
    plt.ylim([0, 1000])
    plt.xlim([0, 60])
    plt.savefig(ALLY_PATH1 + str(i).zfill(3))
    fig.clf()
    plt.close()
files = glob.glob(ALLY_PATH1 + '*.png')
files = sorted(files)
print(files)
for filename in files:
    ally_images1.append(imageio.imread(filename))
imageio.mimsave('./ally_rate.gif', ally_images1)
