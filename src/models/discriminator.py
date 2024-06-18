import torch.nn as nn
import torch


class DiscriminatorCA(nn.Module):
    """
    Discrimnator Conditioned with action.

    """

    def __init__(self, site_fm=64, channel_size=3, action_size=7):
        super(DiscriminatorCA, self).__init__()
        self.action_size = action_size  # Store the number of action dimensions
        self.main = nn.Sequential(
            nn.Conv2d(channel_size + action_size, site_fm, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(site_fm, site_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(site_fm * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(site_fm * 2, site_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(site_fm * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(site_fm * 4, site_fm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(site_fm * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(site_fm * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, action):
        # Expand and concatenate action tensor
        b, c, h, w = x.size()
        action_expanded = action.view(b, self.action_size, 1, 1).expand(
            b, self.action_size, h, w
        )
        x = torch.cat([x, action_expanded], 1)
        # Pass through the main sequential layers
        return self.main(x)


class DiscriminatorCA2(nn.Module):
    """
    Discrimnator Conditioned with action.

    """

    def __init__(self, site_fm=64, channel_size=3, action_size=7):
        super(DiscriminatorCA2, self).__init__()
        self.action_size = action_size  # Store the number of action dimensions
        self.main = nn.Sequential(
            nn.Conv2d(channel_size, site_fm, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(site_fm, site_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(site_fm * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(site_fm * 2, site_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(site_fm * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(site_fm * 4, site_fm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(site_fm * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(site_fm * 8, site_fm * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(site_fm * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Project the action vector to match the feature map dimensions
        self.fc = nn.Linear(action_size, site_fm * 16)
        self.final_conv = nn.Conv2d(site_fm * 16, 1, 2, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, action):
        # Pass the image through the main sequential layers
        x = self.main(x)

        # Flatten the action vector and project it
        action = self.fc(action)
        action = action.view(action.size(0), -1, 1, 1)
        action = action.expand_as(x)

        # Combine the feature maps with the projected action vector
        x = x + action
        # Final convolution and sigmoid
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, site_fm=64, size_z=100, channel_size=3):
        super(Discriminator, self).__init__()
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
