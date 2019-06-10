import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from Dataset import Dataset
from torch.optim import Adam, RMSprop
import math
from torch.autograd import Variable

import imageio

# Machine learning parameters
NUM_EPOCHS = 300
BATCH_SIZE = 256

# Comms parameters
CHANNEL_USE = 7 # The parameter n
BLOCK_SIZE = 4 # The parameter k

# Torch parameters
USE_CUDA = True 

class Net(nn.Module):
    def __init__(self, in_channels, compressed_dim, snr):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.compressed_dim = compressed_dim
        self.snr = snr
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels, compressed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        ) 

    def decode_signal(self, x):
        return self.decoder(x)

    def encode(self, x):
        x = self.encoder(x)

        # Normalization so that every element of x is normalized. Forces unit amplitude...? question about coding vs modulation
        #x = (x / x.norm(dim=-1)[:, None])

        # Normalization. Forces average power to be equal to n.
        #x = x / CHANNEL_USE
        x = x / x.norm() * 70


        return x

    def forward(self, x):
        x = self.encode(x)

        # 7dBW to SNR.
        training_signal_noise_ratio = 10**(0.1*self.snr) 

        # bit / channel_use
        communication_rate = BLOCK_SIZE / CHANNEL_USE   

        # Simulated Gaussian noise.
        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        if USE_CUDA: noise = noise.cuda()
        x += noise

        x = self.decoder(x)

        return x

def accuracy(preds, labels):
    return torch.sum(torch.eq(pred, labels)).item()/(list(preds.size())[0])


if __name__ == "__main__":
    train_new_model = True 
    if train_new_model == True:
        # Data generation
        train_labels = (torch.rand(10000) * (2**BLOCK_SIZE)).long()
        train_data = torch.eye(2**BLOCK_SIZE).index_select(dim=0, index=train_labels)
        test_labels = (torch.rand(1500) * (2**BLOCK_SIZE)).long()
        test_data = torch.eye(2**BLOCK_SIZE).index_select(dim=0, index=test_labels)

        # Data loading
        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 6}
        training_set = Dataset(train_data, train_labels)
        training_loader = torch.utils.data.DataLoader(training_set, **params)
        loss_fn = nn.CrossEntropyLoss()

        # Training
        snrs_db = np.linspace(-4, 9, num=14)
        for snr in snrs_db:
            loss_list = []
            acc_list = []

            # The input dimension to the autoencoder should be the number of 
            # possible codewords, i.e. 2^k. This is because k represents the
            # number of bits of the block we wish to transmit, and we are 
            # representing input as a one-hot vector, so each input vector
            # will be of length 2^k.
            # The hidden dimension to the autoencoder should be n, the channel
            # use. This is because we wish the autoencoder to learn an optimal
            # coding scheme, so the middle layer should encode a vector of length
            # n.
            model = Net(2**BLOCK_SIZE, compressed_dim=CHANNEL_USE, snr=snr)
            if USE_CUDA: model = model.cuda()
            optimizer = Adam(model.parameters(), lr=0.001)

            #with imageio.get_writer('results/gifs/normalization.gif', mode='I') as writer:
            for epoch in range(NUM_EPOCHS): 
                for batch_idx, (batch, labels) in enumerate(training_loader):
                    if USE_CUDA:
                        batch = batch.cuda()
                        labels = labels.cuda()
                    output = model(batch)
                    loss = loss_fn(output, labels)

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch_idx % (BATCH_SIZE-1) == 0:
                        pred = torch.argmax(output, dim=1)
                        acc = accuracy(pred, labels)  
                        print('Epoch %2d for SNR %s: loss=%.4f, acc=%.2f' % (epoch, snr, loss.item(), acc))
                        loss_list.append(loss.item())
                        acc_list.append(acc)

                        train_data = train_data.cuda()
                        train_codes = model.encode(train_data)
                        train_codes = train_codes.cpu().detach().numpy()
                        fig = plt.figure()
                        plt.scatter(train_codes[:, 0], train_codes[:, 1])
                        plt.savefig('results/images/bler_plot'+str(epoch)+'.png')
                        fig.clf()
                        plt.close()
                        image = imageio.imread('results/images/bler_plot'+str(epoch)+'.png')
                        #writer.append_data(image)


                torch.save(model.state_dict(), './models/model_state_for_recreating_bler_plot'+str(snr))

    else:
        test_labels = (torch.rand(1500) * CHANNEL_SIZE).long()
        test_data = torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)
        model = Net(CHANNEL_SIZE, compressed_dim=int(math.log2(CHANNEL_SIZE)), snr=2)
        model.load_state_dict(torch.load('./model_state'))
        model = model.cuda()
        test_data = test_data.cuda()
        test_codes = model.encode(test_data)
        test_codes = test_codes.cpu().detach().numpy()
        print(test_codes)
        plt.scatter(test_codes[:, 0], test_codes[:, 1])
        plt.savefig('foo.png')
