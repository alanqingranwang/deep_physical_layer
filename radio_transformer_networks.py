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

NUM_EPOCHS = 100
BATCH_SIZE = 256
CHANNEL_SIZE = 4
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
            nn.Linear(compressed_dim, compressed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(compressed_dim, in_channels)
        ) 

    def decode_signal(self, x):
        return self.decoder(x)

    def encode(self, x):
        x = self.encoder(x)
        # Normalization.
        x = (x / x.norm(dim=-1)[:, None])
        return x

    def forward(self, x):
        x = self.encode(x)

        # 7dBW to SNR.
        training_signal_noise_ratio = 10**(0.1*self.snr) 

        # bit / channel_use
        communication_rate = 1   

        # Simulated Gaussian noise.
        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        if USE_CUDA: noise = noise.cuda()
        x += noise

        x = self.decoder(x)

        return x

def accuracy(preds, labels):
     return torch.sum(torch.eq(pred, labels)).item()/BATCH_SIZE


if __name__ == "__main__":
    train_new_model = True 
    if train_new_model == True:
        train_labels = (torch.rand(10000) * CHANNEL_SIZE).long()
        train_data = torch.eye(CHANNEL_SIZE).index_select(dim=0, index=train_labels)
        test_labels = (torch.rand(1500) * CHANNEL_SIZE).long()
        test_data = torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)

        # Parameters
        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 6}
        training_set = Dataset(train_data, train_labels)
        training_loader = torch.utils.data.DataLoader(training_set, **params)
        loss_fn = nn.CrossEntropyLoss()

        snrs_db = np.linspace(-4, 9, num=14)
        snrs_db = [2]
        for snr in snrs_db:
            loss_list = []
            acc_list = []

            #model = Net(CHANNEL_SIZE, compressed_dim=int(math.log2(CHANNEL_SIZE)), snr=snr)
            model = Net(CHANNEL_SIZE, compressed_dim=2, snr=snr)
            if USE_CUDA: model = model.cuda()
            optimizer = Adam(model.parameters(), lr=0.001)

            with imageio.get_writer('images/gif/normalization.gif', mode='I') as writer:
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
                            plt.savefig('images/foo'+str(epoch)+'.png')
                            fig.clf()
                            plt.close()
                            image = imageio.imread('images/foo'+str(epoch)+'.png')
                            writer.append_data(image)


                #torch.save(model.state_dict(), './models/model_state_'+str(snr))

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
