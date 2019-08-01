Communications modeling with deep neural networks.  

Main branch contains autoencoder and RNN code. 
Test branch contains unstable GAN code for channel modeling.  

## Usage 
python main.py -h

### Example
python main.py --autoencoder --use_complex --channel_type 'awgn' --channel_use 32 --block_size 4 --epochs 300 --snr -5 --no_lpf

## Known bugs
+ Complex numbers not integrated for RNN models
