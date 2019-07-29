import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, gen_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(gen_dim, gen_dim, normalize=False),
            *block(gen_dim, gen_dim),
            # *block(gen_dim, gen_dim),
            # *block(gen_dim, gen_dim),
            nn.Linear(gen_dim, gen_dim),
            nn.Sigmoid()
        )

    def forward(self, input_codeword):
        img = self.model(input_codeword)
        return img


class Discriminator(nn.Module):
    def __init__(self, gen_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(gen_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(32, 16),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, real_or_fake_data):
        return self.model(real_or_fake_data)

    def accuracy(self, real_data, fake_data, cuda):
        ones = torch.ones(real_data.size())
        zeros = torch.zeros(fake_data.size())
        if cuda:
            ones = ones.cuda()
            zeros = zeros.cuda()

        num_diff1 = torch.sum(torch.abs(self.forward(real_data) - ones))
        num_diff2 = torch.sum(torch.abs(self.forward(fake_data) - zeros))
        return 1-(num_diff1 + num_diff2) / (real_data.size(0)*real_data.size(1) + fake_data.size(0)*fake_data.size(1))
