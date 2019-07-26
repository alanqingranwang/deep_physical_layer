#!/bin/bash

python experimental.py --use_complex --no_lpf --block_size 4 --channel_use 4 --epochs 100 --snr -5
python experimental.py --use_complex --no_lpf --block_size 4 --channel_use 8 --epochs 100 --snr -5
