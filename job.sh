#!/bin/bash

for i in $(seq -5 1 10)
do
    python experimental.py --use_complex --no_lpf --channel_use 32 --epochs 500 --snr $i
done
