import torch
import torch.nn as nn


class ResidualBlockWithImage(nn.Module):
    def __init__(self, in_features, state_features):
        super(ResidualBlockWithImage, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features + state_features, in_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x, image, action):
        # Concatenate the current image and action along the channel dimension
        action_expanded = action.view(action.size(0), action.size(1), 1, 1).repeat(
            1, 1, x.size(2), x.size(3)
        )
        image_action = torch.cat((image, action_expanded), dim=1)
        x = torch.cat((x, image_action), dim=1)
        return x + self.block(x)


class ResidualBlockCombineImage(nn.Module):
    def __init__(self, in_features, image_channels):
        super(ResidualBlockCombineImage, self).__init__()
        self.image_downsample = nn.Sequential(
            nn.Conv2d(
                image_channels,
                in_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_features * 2, in_features * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_features * 2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(in_features),
        )

    def forward(self, x, image):
        # Downsample the image to match the feature map dimensions
        image_downsampled = self.image_downsample(image)
        # Concatenate the input feature map x with the downsampled image
        x = torch.cat((x, image_downsampled), dim=1)
        return x + self.block(x)


class GeneratorActorRes(nn.Module):
    def __init__(self, size_fm=64, size_z=100, channel_size=3, state_features=11):
        super(GeneratorActorRes, self).__init__()
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(
                size_z + state_features + channel_size, size_fm * 8, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(size_fm * 8),
            nn.ReLU(True),
        )
        self.res_block1 = ResidualBlockWithImage(
            size_fm * 8, state_features + channel_size
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(size_fm * 8, size_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 4),
            nn.ReLU(True),
        )
        self.res_block2 = ResidualBlockWithImage(
            size_fm * 4, state_features + channel_size
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(size_fm * 4, size_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 2),
            nn.ReLU(True),
        )
        self.res_block3 = ResidualBlockWithImage(
            size_fm * 2, state_features + channel_size
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(size_fm * 2, size_fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm),
            nn.ReLU(True),
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(size_fm, channel_size, 4, 2, 1, bias=False), nn.Tanh()
        )

    def forward(self, noise, image, action):
        # Encode angles using sine and cosine
        sin_cos = torch.cat(
            [torch.sin(action[:, 3:]), torch.cos(action[:, 3:])], dim=-1
        )

        # Concatenate positions and sine-cosine encoded angles
        action = torch.cat([action[:, :3], sin_cos], dim=-1)

        # Reshape action to make it compatible for concatenation
        action = action.view(action.size(0), action.size(1), 1, 1)

        # # Repeat the action tensor to match input dimensions
        # action = action.repeat(1, 1, noise.size(2), noise.size(3))
        print(image.size())
        print(action.size())
        # Concatenate noise, current image, and action
        image_action = torch.cat((image, action), dim=1)
        input = torch.cat((noise, image_action), 1)

        # Forward through the initial layers and residual blocks with state
        x = self.initial(input)
        x = self.res_block1(x, image, action)
        x = self.upsample1(x)
        x = self.res_block2(x, image, action)
        x = self.upsample2(x)
        x = self.res_block3(x, image, action)
        x = self.upsample3(x)
        return self.final(x)


class GeneratorActorResIm(nn.Module):
    def __init__(self, size_fm=64, size_z=100, channel_size=3, action_size=7):
        super(GeneratorActorResIm, self).__init__()

        # Initial layers operating on noise and actions
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(
                size_z + action_size + 4, size_fm * 8, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(size_fm * 8),
            nn.ReLU(True),
        )

        # Residual blocks integrating the current image
        self.res_block1 = ResidualBlockCombineImage(size_fm * 8, channel_size)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(1024, size_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 4),
            nn.ReLU(True),
        )
        self.res_block2 = ResidualBlockCombineImage(size_fm * 4, channel_size)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(size_fm * 4, size_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm * 2),
            nn.ReLU(True),
        )
        self.res_block3 = ResidualBlockCombineImage(size_fm * 2, channel_size)
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(size_fm * 2, size_fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_fm),
            nn.ReLU(True),
        )

        # Final layer predicting the next image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(size_fm, channel_size, 4, 2, 1, bias=False), nn.Tanh()
        )

    def forward(self, noise, action, image):
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
        input_combined = torch.cat((noise, action), 1)

        # Forward through the initial layers
        x = self.initial(input_combined)
        # Integrate the current image in the residual blocks
        x = self.res_block1(x, image)

        x = self.upsample1(x)
        # x = self.res_block2(x, image)
        x = self.upsample2(x)
        # x = self.res_block3(x, image)
        x = self.upsample3(x)

        # Predict the next image frame
        return self.final(x)
