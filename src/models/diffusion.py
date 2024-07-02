import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
from transformers import ViTModel, ViTConfig


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels):
        super(UNet, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.decoder1 = self.up_conv_block(1024, 256)
        self.decoder2 = self.up_conv_block(256 + 256, 128)
        self.decoder3 = self.up_conv_block(128 + 128, 64)
        self.decoder4 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

        self.conditioning_layer = nn.Linear(cond_channels, 512)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernels_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, cond):
        # Encode
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))

        # Apply conditioning
        cond = self.conditioning_layer(cond)
        cond = cond.view(cond.size(0), -1, 1, 1)
        cond = cond.expand(cond.size(0), -1, e4.size(2), e4.size(3))

        e4 = torch.cat([e4, cond], dim=1)

        # Decode
        d3 = self.decoder1(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d1 = self.decoder3(d2)
        d1 = torch.cat([d1, e1], dim=1)
        out = self.decoder4(d1)

        return out


class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels):
        super(UNet2, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.decoder1 = self.up_conv_block(512 + 512, 256)
        self.decoder2 = self.up_conv_block(256 + 256, 128)
        self.decoder3 = self.up_conv_block(128 + 128, 64)
        self.decoder4 = nn.Conv2d(64 + 64, out_channels, kernel_size=3, padding=1)

        self.conditioning_layer = nn.Sequential(
            nn.Linear(cond_channels, 512), nn.ReLU(inplace=True)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Consider adding dropout for regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, cond):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))

        # Apply conditioning
        cond = self.conditioning_layer(cond)
        cond = cond.view(cond.size(0), -1, 1, 1)
        cond = cond.expand(cond.size(0), -1, e4.size(2), e4.size(3))

        e4 = torch.cat([e4, cond], dim=1)

        # Decoder with skip connections
        d3 = self.decoder1(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d1 = self.decoder3(d2)
        d1 = torch.cat([d1, e1], dim=1)
        out = self.decoder4(d1)

        return out


class DiffusionForward:

    def __init__(self, betas, device, save_dir="forward_debug"):
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.save_dir = save_dir
        self.device = device
        os.makedirs(self.save_dir, exist_ok=True)

    def add_noise(self, x, t, save_image=False, image_id=0, epoch=0):
        """Add noise to the images based on the schedule."""
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t][
            :, None, None, None
        ]  # Broadcasting to image dimensions
        noisy_image = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise
        if save_image:
            self.save_image(
                noisy_image,
                f"{self.save_dir}/noisy_epoch_{epoch}_id_{image_id}.png",
            )
        noisy_image = noisy_image.to(dtype=torch.float32, device=self.device)
        return noisy_image, noise

    def save_image(self, tensor_image, filename):
        image = tensor_image[0].permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save(filename)


class DiffusionReverse:
    def __init__(self, betas, model, save_dir="reverse_debug"):

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.model = model
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def remove_noise(self, noisy_image, timestep, cond):
        """Predict and remove noise from the noisy image."""
        batch_size = noisy_image.size(0)

        predicted_noise = self.model(noisy_image, cond)

        alpha_t = self.alphas_cumprod[timestep]
        beta_t = self.betas[timestep]

        # Reshape to [batch_size, 1, 1, 1] for broadcasting
        alpha_t = alpha_t.view(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)
        beta_t = beta_t.view(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)

        denoised_image = (1 / alpha_t.sqrt()) * (
            noisy_image - beta_t * predicted_noise / (1 - alpha_t).sqrt()
        )
        return denoised_image

    def reverse_diffusion(
        self, noisy_image, cond, save_image=False, image_id=0, epoch=0
    ):
        """Iterate through timesteps to denoise the image."""
        for timestep in reversed(range(len(self.betas))):
            noisy_image = self.remove_noise(noisy_image, timestep, cond)

        if save_image:
            self.save_image(
                noisy_image,
                f"{self.save_dir}/epoch_{epoch}_timestep_{timestep}_id_{image_id}.png",
            )
        return noisy_image

    def save_image(self, tensor_image, filename):
        with torch.no_grad():
            image_np = tensor_image[0].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        Image.fromarray(image_np).save(filename)


class TransformerDiffusionModel(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, cond_channels):
        super(TransformerDiffusionModel, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=image_size,
            patch_size=patch_size,
            num_labels=num_patches,
        )
        self.transformer = ViTModel(config)
        self.to_logits = nn.Linear(config.hidden_size, 24576)

        self.conditioning_layer = nn.Linear(cond_channels, 3)
        self.up_conv = nn.ConvTranspose2d(408, num_classes, kernel_size=2, stride=2)
        # self.up_conv2 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, x, cond):

        cond = self.conditioning_layer(cond).unsqueeze(2).unsqueeze(3)
        cond = cond.expand(-1, -1, x.size(2), x.size(3))

        x = x + cond
        outputs = self.transformer(pixel_values=x)
        sequence_output = outputs.last_hidden_state
        logits = self.to_logits(sequence_output[:, :, :])
        logits = logits.view(x.size()[0], -1, self.patch_size * 2, self.patch_size * 2)
        logits = self.up_conv(logits)
        # logits = self.up_conv2(logits)
        # print(logits.size())
        return logits
