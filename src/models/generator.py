import torch.nn as nn
import torch


class GeneratorActor(nn.Module):

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


class GeneratorAttention(nn.Module):
    def __init__(self, nz=100, na=7, size_fm=64, nc=3):
        super(GeneratorAttention, self).__init__()
        self.nz = nz
        self.na = na
        self.size_fm = size_fm

        # Process noise z
        self.noise_fc = nn.Sequential(
            nn.Linear(nz, self.size_fm * 8 * 8 * 8), nn.ReLU(True)
        )

        # Attention mechanism for actions
        self.attn_fc = nn.Linear(na, size_fm * 8 * 8 * 8)
        self.attn = nn.MultiheadAttention(embed_dim=size_fm * 8 * 8 * 8, num_heads=8)

        # Convolutional layers
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(
                self.size_fm * 8 * 2, self.size_fm * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.size_fm * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.size_fm * 4, self.size_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.size_fm * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.size_fm * 2, self.size_fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.size_fm),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.size_fm, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, action):
        # Process noise
        noise_output = self.noise_fc(z)
        noise_output = noise_output.view(noise_output.size(0), self.size_fm * 8, 8, 8)

        # Process action through attention
        action_proj = (
            self.attn_fc(action)
            .view(action.size(0), self.size_fm * 8, 8 * 8)
            .permute(2, 0, 1)
        )
        print(action_proj.size())
        action_attn, _ = self.attn(action_proj, action_proj, action_proj)
        action_attn = action_attn.permute(1, 2, 0).view(
            action.size(0), self.size_fm * 8, 8, 8
        )

        print(action_attn.size())
        print(noise_output.size())
        print(action_attn.size())
        # Concatenate noise and action output
        combined = torch.cat([noise_output, action_attn], dim=1)

        # Generate image
        return self.gen(combined)


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
