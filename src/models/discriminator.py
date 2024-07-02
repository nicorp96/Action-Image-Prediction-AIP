import torch.nn as nn
import torch


class DiscriminatorCA(nn.Module):
    """
    Discrimnator Conditioned with action.
    this architecture is based on the DCGAN principles but includes additional conditioning information in the form of action inputs.
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


class DiscriminatorCA1(nn.Module):
    """
    Discrimnator Conditioned with action.
    this architecture is based on the DCGAN principles but includes additional conditioning information in the form of action inputs.
    """

    def __init__(self, site_fm=64, channel_size=3, action_size=7):
        super(DiscriminatorCA1, self).__init__()
        self.action_size = action_size  # Store the number of action dimensions
        self.fc = nn.Sequential(
            nn.Linear(action_size, site_fm * site_fm * channel_size), nn.ReLU()
        )
        self.main = nn.Sequential(
            nn.Conv2d(channel_size * 2, site_fm, 4, 2, 1, bias=False),
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
        action_expanded = self.fc(action).view(x.size())
        x = torch.cat([x, action_expanded], 1)
        # Pass through the main sequential layers
        return self.main(x)


class DiscriminatorCA2(nn.Module):
    """
    Discrimnator Conditioned with action on last layer.
    This architecture is based on the DCGAN principles but includes additional conditioning information in the form of action inputs.
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
        self.final_conv = nn.Sequential(
            nn.Conv2d(site_fm * 16, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 2, 2, 1, bias=False),
        )
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
        x = x.view(-1, 1)  # Reshape to (N, 1)
        x = self.sigmoid(x)
        return x


class DiscriminatorAuxAct(nn.Module):
    """
    This architecture is based on the principles of the Auxiliary Classifier GAN (AC-GAN).
    Instead of classifying discrete classes, it regresses actions.
    """

    def __init__(self, site_fm=64, action_size=7, channel_size=3):
        super(DiscriminatorAuxAct, self).__init__()
        self.action_size = action_size  # Store the number of action dimensions

        self.shared_conv = nn.Sequential(
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
        )

        # Real/Fake prediction branch
        self.real_fake_branch = nn.Sequential(
            nn.Conv2d(site_fm * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        # Action coordinates regression branch
        self.action_branch = nn.Sequential(
            nn.Conv2d(site_fm * 8, site_fm * 4, 2, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(site_fm * 4, site_fm * 2, 2, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(
                site_fm * 2 * 2 * 2, action_size
            ),  # Adjust based on spatial dimensions
        )

    def forward(self, x):
        # Pass through the shared convolutional layers
        features = self.shared_conv(x)

        # Branch into real/fake and action coordinate predictions
        real_fake_output = self.real_fake_branch(features).view(-1, 1)
        action_output = self.action_branch(features)

        return real_fake_output, action_output


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


class DiscriminatorFK(nn.Module):
    def __init__(self, image_size, channel_size=3, action_size=7):
        super(DiscriminatorFK, self).__init__()
        self.action_size = action_size
        self.model = nn.Sequential(
            nn.Conv2d(
                channel_size + action_size,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(
                512 * (image_size // 32) * (image_size // 32), 1
            ),  # Adjust size if needed
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
        return self.model(x)
