import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, img_channels, feature_dim, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            img_channels, feature_dim, kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1
        )

        self.fc1 = nn.Linear(feature_dim * 4 * 8 * 8, z_dim)  # Mean
        self.fc2 = nn.Linear(feature_dim * 4 * 8 * 8, z_dim)  # Log Variance

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        mean = self.fc1(x)
        log_var = self.fc2(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, z_dim, feature_dim, img_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_dim, feature_dim * 4 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(
            feature_dim * 4, feature_dim * 2, kernel_size=4, stride=2, padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            feature_dim * 2, feature_dim, kernel_size=4, stride=2, padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            feature_dim, img_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), -1, 8, 8)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(
            self.deconv3(x)
        )  # Use sigmoid to get pixel values in the range [0, 1]
        return x


class VAE(nn.Module):
    def __init__(self, img_channels=3, feature_dim=64, z_dim=100):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, feature_dim, z_dim)
        self.decoder = Decoder(z_dim, feature_dim, img_channels)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Random normal variable
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decoder(z)
        return recon_x, mean, log_var
