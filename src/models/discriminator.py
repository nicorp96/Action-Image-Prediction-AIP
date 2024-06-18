import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.main(x)
        output = output.view(x.size(0), -1)
        return output.mean(dim=1)


class Discriminator2(nn.Module):

    def __init__(self, site_fm=64, size_z=100, channel_size=3):
        super(Discriminator2, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(channel_size, site_fm, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(site_fm) x 32 x 32``
            nn.Conv2d(site_fm, site_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(site_fm * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(site_fm*2) x 16 x 16``
            nn.Conv2d(site_fm * 2, site_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(site_fm * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(site_fm*4) x 8 x 8``
            nn.Conv2d(site_fm * 4, site_fm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(site_fm * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(site_fm*8) x 4 x 4``
            nn.Conv2d(site_fm * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
