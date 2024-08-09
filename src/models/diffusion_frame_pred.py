import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed
from einops import rearrange, repeat
from .difussion_utils.transformers_utils import (
    DiTBlock,
    modulate,
    get_2d_sincos_pos_embed,
    get_1d_sincos_temp_embed,
)

from .embedders import TimestepEmbedder, ActionEmbedder


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTActionSeq(nn.Module):

    def __init__(self, config):
        super(DiTActionSeq, self).__init__()
        self.input_size = config["input_size"]
        self.patch_size = config["patch_size"]
        self.in_channels = config["in_channels"]
        self.hidden_size = config["hidden_size"]
        self.depth = config["depth"]
        self.num_heads = config["num_heads"]
        self.mlp_ratio = config["mlp_ratio"]
        self.action_dim = config["action_dim"]
        self.learn_sigma = config["learn_sigma"]
        self.seq_len = config["seq_len"]
        self.mask_n = config["mask_n"]
        self.out_channels = self.in_channels * 2 if self.learn_sigma else 4

        self.x_embedder = PatchEmbed(
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bias=True,
        )
        self.a_embedder = ActionEmbedder(self.hidden_size, self.action_dim)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
                for _ in range(self.depth)
            ]
        )
        self.final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize action embedding MLP: Added this part
        nn.init.normal_(self.a_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.a_embedder.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    # TODO: Add current image as embedding https://github.com/homangab/Track-2-Act/blob/main/single_script.py
    def forward(self, x, t, a, mask_frame_num=None):
        """
        Forward pass of DiT which now also takes actions as input
        x: (N, L, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        a: (N, C, 7) tensor of actions (TCP positions in (x, y, z, rpx, rpy, rpz))
        """
        batch_sz, l, ch, h, w = x.shape

        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed
        a = rearrange(a, "b f d -> (b f) d")
        a = self.a_embedder(a)  # (N, D) Action embedding
        t = self.t_embedder(t)  # (N, D)
        timestep_spatial = repeat(t, "n d -> (n c) d", c=l)
        c = timestep_spatial + a  # + y_emb
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=batch_sz)
        return x


class DiTActionSeqISim(DiTActionSeq):
    # https://github.com/bytedance/IRASim/blob/main/models/irasim.py
    def __init__(self, config):
        super().__init__(config)
        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.hidden_size), requires_grad=False
        )
        temp_embed = get_1d_sincos_temp_embed(
            self.temp_embed.shape[-1], self.temp_embed.shape[-2]
        )
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

    def forward(self, x, t, a, mask_frame_num=None):
        """
        Forward pass of DiT which now also takes actions as input
        x: (N, L, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        a: (N, C, 7) tensor of actions (TCP positions in (x, y, z, rpx, rpy, rpz))
        """
        batch_sz, l, ch, h, w = x.shape

        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed
        a = rearrange(a, "b f d -> (b f) d")
        a = self.a_embedder(a)  # (N, D) Action embedding
        t = self.t_embedder(t)  # (N, D)

        timestep_spatial = repeat(t, "n d -> (n c) d", c=l)
        timestep_temp = repeat(t, "n d -> (n c) d", c=self.pos_embed.shape[1])

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]
            c = timestep_spatial + a
            x = spatial_block(x, c)
            x = rearrange(x, "(b f) t d -> (b t) f d", b=batch_sz)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed[:, 0:l]
            c = timestep_temp
            x = temp_block(x, c)
            x = rearrange(x, "(b t) f d -> (b f) t d", b=batch_sz)

        c = timestep_spatial + a
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=batch_sz)
        return x


class ConditionEmbedding(nn.Module):

    def __init__(self, out_dim=1200):
        super(ConditionEmbedding, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=out_dim, kernel_size=4, stride=2
        )

        # ReLU activation
        self.relu = nn.ReLU()

        # Initialize weights with Gaussian distribution
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x


class DiTActionSeqISimMultiCondi(DiTActionSeq):
    # https://arxiv.org/pdf/2302.05543
    def __init__(self, config):
        super().__init__(config)
        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.hidden_size), requires_grad=False
        )
        temp_embed = get_1d_sincos_temp_embed(
            self.temp_embed.shape[-1], self.temp_embed.shape[-2]
        )
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))
        self.condition_emb = ConditionEmbedding(out_dim=96 * 5)

    def forward(self, x, t, a, c_m, mask_frame_num=None):
        """
        Forward pass of DiT which now also takes actions as input
        x: (N, L, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        a: (N, C, 7) tensor of actions (TCP positions in (x, y, z, rpx, rpy, rpz))
        """
        batch_sz, l, ch, h, w = x.shape

        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed
        a = rearrange(a, "b f d -> (b f) d")
        a = self.a_embedder(a)  # (N, D) Action embedding
        t = self.t_embedder(t)  # (N, D)
        c_m = rearrange(c_m, "b l c w h-> (b l) c w h")  # [32*2, 3, 64, 64]
        canny_emb = self.condition_emb(c_m)
        canny_emb = canny_emb.flatten()
        canny_emb = rearrange(canny_emb, "(b h) -> b h", b=a.shape[0])
        timestep_spatial = repeat(t, "n d -> (n c) d", c=l)
        timestep_temp = repeat(t, "n d -> (n c) d", c=self.pos_embed.shape[1])

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]
            c = timestep_spatial + a + canny_emb
            x = spatial_block(x, c)
            x = rearrange(x, "(b f) t d -> (b t) f d", b=batch_sz)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed[:, 0:l]
            c = timestep_temp
            x = temp_block(x, c)
            x = rearrange(x, "(b t) f d -> (b f) t d", b=batch_sz)

        c = timestep_spatial + a
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=batch_sz)
        return x
