'''
Alan Wang, AL29162
7/31/19

This script produces gifs from pictures saved in AWGN_PATH
'''
import imageio
import matplotlib.pyplot as plt
import os
import glob
import itertools
import torch
from experimental import Net

AWGN_PATH = '../awgn_pics/'
awgn_images = []

files = glob.glob(AWGN_PATH + '*.png')
files = sorted(files)
print(files)
for filename in files:
    awgn_images.append(imageio.imread(filename))
imageio.mimsave('./results/gifs/awgn.gif', awgn_images)
