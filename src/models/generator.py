import torch.nn as nn
import torch
import numpy as np


class GeneratorActor(nn.Module):
    """GAN based on Deep Convolutional GAN (DCGAN)
    Transposed Convolutions: Perform upsampling to generate a larger image from the latent vector.
    """

    def __init__(self, size_fm=64, size_z=100, channel_size=3):
        super(GeneratorActor, self).__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(size_z + 11, size_fm * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(size_fm * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 8, size_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 4, size_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 2, size_fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm, channel_size, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, action):
        # Encode angles using sine and cosine
        sin_cos = torch.cat(
            [torch.sin(action[:, 3:]), torch.cos(action[:, 3:])], dim=-1
        )

        # Concatenate positions and sine-cosine encoded angles
        action = torch.cat([action[:, :3], sin_cos], dim=-1)

        # Reshape action to make it compatible for concatenation
        action = action.view(action.size(0), action.size(1), 1, 1)

        # Repeat the action tensor to match input dimensions
        action = action.repeat(1, 1, noise.size(2), noise.size(3))

        # Concatenate input and action
        input = torch.cat((noise, action), 1)

        return self.seq(input)


class GeneratorActor2(nn.Module):

    def __init__(self, size_fm=64, size_z=100, channel_size=3):
        super(GeneratorActor2, self).__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(size_z + 11, size_fm * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(size_fm * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 16, size_fm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 8, size_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 4, size_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 2, size_fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm, channel_size, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, action):
        # Encode angles using sine and cosine
        sin_cos = torch.cat(
            [torch.sin(action[:, 3:]), torch.cos(action[:, 3:])], dim=-1
        )

        # Concatenate positions and sine-cosine encoded angles
        action = torch.cat([action[:, :3], sin_cos], dim=-1)

        # Reshape action to make it compatible for concatenation
        action = action.view(action.size(0), action.size(1), 1, 1)

        # Repeat the action tensor to match input dimensions
        action = action.repeat(1, 1, noise.size(2), noise.size(3))

        # Concatenate input and action
        input = torch.cat((noise, action), 1)

        return self.seq(input)


class GeneratorActorUN(nn.Module):

    def __init__(self, size_fm=64, size_z=100, channel_size=3):
        super(GeneratorActorUN, self).__init__()

        self.downsampler = nn.Sequential(
            nn.Conv2d(size_z + 11, size_fm * 4, 2, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 4),
            nn.ReLU(True),
            nn.Conv2d(size_fm * 4, size_fm * 8, 2, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 8),
            nn.ReLU(True),
            nn.Conv2d(size_fm * 8, size_fm * 16, 2, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 16),
            nn.ReLU(True),
        )

        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(size_fm * 16, size_fm * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(size_fm * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 8, size_fm * 4, 2, 1, 0, bias=False),
            nn.BatchNorm2d(size_fm * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 4, size_fm * 2, 2, 1, 0, bias=False),
            nn.BatchNorm2d(size_fm * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 2, size_fm, 2, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm, channel_size, 2, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, action):
        # Encode angles using sine and cosine
        sin_cos = torch.cat(
            [torch.sin(action[:, 3:]), torch.cos(action[:, 3:])], dim=-1
        )

        # Concatenate positions and sine-cosine encoded angles
        action = torch.cat([action[:, :3], sin_cos], dim=-1)

        # Reshape action to make it compatible for concatenation
        action = action.view(action.size(0), action.size(1), 1, 1)

        # Repeat the action tensor to match input dimensions
        action = action.repeat(1, 1, noise.size(2), noise.size(3))

        # Concatenate input and action
        input = torch.cat((noise, action), 1)

        # Downsample
        downsampled = self.downsampler(input)

        # Upsample and output
        return self.upsampler(downsampled)


class Generator(nn.Module):
    def __init__(self, size_fm=64, size_z=100, channel_size=3):
        super(Generator, self).__init__()

        self.seq = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(size_z, size_fm * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(size_fm * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(size_fm * 8, size_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(size_fm * 4, size_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 2, size_fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(size_fm, channel_size, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.seq(input)


class ImageProcessor(nn.Module):
    """CNN module to process the input robot image"""

    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(512 * 4 * 4, 100)

    def forward(self, image):
        x = self.conv_layers(image)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class GeneratorActorImg(nn.Module):

    def __init__(self, size_fm=64, size_z=100, channel_size=3):
        super(GeneratorActorImg, self).__init__()
        self.image_processor = ImageProcessor()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(size_z + 100 + 11, size_fm * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(size_fm * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 8, size_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 4, size_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm * 2, size_fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm),
            nn.ReLU(True),
            nn.ConvTranspose2d(size_fm, channel_size, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, action, current_image):
        # Process the image to get a feature vector
        image_features = self.image_processor(current_image)

        # Encode angles using sine and cosine
        sin_cos = torch.cat(
            [torch.sin(action[:, 3:]), torch.cos(action[:, 3:])], dim=-1
        )

        # Concatenate positions and sine-cosine encoded angles
        action = torch.cat([action[:, :3], sin_cos], dim=-1)

        # Reshape action to make it compatible for concatenation
        action = action.view(action.size(0), action.size(1), 1, 1)

        # Repeat the action tensor to match input dimensions
        action = action.repeat(1, 1, noise.size(2), noise.size(3))

        # Concatenate noise, action and image features
        combined_input = torch.cat((noise, action), 1)

        # Repeat the reshaped image_features tensor to match input dimensions
        image_features = image_features.view(image_features.size(0), 100, 1, 1)
        image_features = image_features.repeat(
            1, 1, combined_input.size(2), combined_input.size(3)
        )

        # Final combined input
        input = torch.cat((combined_input, image_features), 1)

        return self.seq(input)


class GeneratorFK(nn.Module):

    def __init__(self, size_z=100, action_dim=11):
        super(GeneratorFK, self).__init__()

        self.size_z = size_z
        self.action_dim = action_dim
        self.linear = nn.Sequential(
            nn.Linear(111, 128 * 128 * 3, bias=False),
        )
        self.seq = nn.Sequential(
            nn.Conv2d(
                3,
                128,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                256, 512, kernel_size=4, stride=1, padding=1, bias=False
            ),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                512, 512, kernel_size=4, stride=1, padding=1, bias=False
            ),
            nn.ConvTranspose2d(
                512, 512, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=1, padding=1, bias=False
            ),
            nn.ConvTranspose2d(
                256, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, action):
        # Encode angles using sine and cosine
        sin_cos = torch.cat(
            [torch.sin(action[:, 3:]), torch.cos(action[:, 3:])], dim=-1
        )

        # Concatenate positions and sine-cosine encoded angles
        action = torch.cat([action[:, :3], sin_cos], dim=-1)

        # # Reshape action to make it compatible for concatenation
        # action = action.view(action.size(0), action.size(1), 1, 1)

        # # Repeat the action tensor to match input dimensions
        # action = action.repeat(noise.size(2), noise.size(3))

        # Concatenate input and action
        input = torch.cat((noise, action), 1)
        input_ve = self.linear(input)
        input_ve = input_ve.view(input_ve.size(0), 3, 128, 128)

        return self.seq(input_ve)
