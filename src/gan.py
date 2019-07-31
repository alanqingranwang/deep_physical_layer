import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, gen_dim, block_size):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(gen_dim + block_size, gen_dim, normalize=False),
            *block(gen_dim, gen_dim),
            # *block(gen_dim, gen_dim),
            # *block(gen_dim, gen_dim),
            nn.Linear(gen_dim, gen_dim),
            nn.Sigmoid()
        )

    def forward(self, input_codeword, original_msg):
        gen_input = torch.cat((input_codeword, original_msg), -1)
        img = self.model(gen_input)
        return img


class Discriminator(nn.Module):
    def __init__(self, gen_dim, block_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(gen_dim + block_size, 32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(32, 16),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, real_or_fake_data, original_msg):
        disc_input = torch.cat((real_or_fake_data, original_msg), -1)
        img = self.model(disc_input)
        return img

    def accuracy(self, real_data, fake_data, cuda, batch):
        ones = torch.ones(real_data.size())
        zeros = torch.zeros(fake_data.size())
        if cuda:
            ones = ones.cuda()
            zeros = zeros.cuda()

        num_diff1 = torch.sum(torch.abs(self.forward(real_data, batch) - ones))
        num_diff2 = torch.sum(torch.abs(self.forward(fake_data, batch) - zeros))
        return 1-(num_diff1 + num_diff2) / (real_data.size(0)*real_data.size(1) + fake_data.size(0)*fake_data.size(1))
