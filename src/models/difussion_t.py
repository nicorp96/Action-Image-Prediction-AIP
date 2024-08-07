import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Mlp
import math
from einops import rearrange, repeat
from .difussion_utils.transformers_utils import Attention, DiTBlockJoint


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ActionEmbedder(nn.Module):
    """
    Embeds actions (e.g., TCP positions in (x, y, z, rpx, rpy, rpz, g_d)) into vector representations.
    """

    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.action_dim = action_dim

    def forward(self, actions):
        action_emb = self.mlp(actions)
        return action_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attention_mode="math",
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attention_mode=attention_mode,
            **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


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


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=8,
        patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        action_dim=6,  # Added action dimension
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else 3  # in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.a_embedder = ActionEmbedder(
            hidden_size, action_dim
        )  # Added action embedder

        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # TODO: Add current image as embedding https://github.com/homangab/Track-2-Act/blob/main/single_script.py
    def forward(self, x, t, a):
        """
        Forward pass of DiT which now also takes actions as input.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        a: (N, 6) tensor of actions (TCP positions in (x, y, z, rpx, rpy, rpz))
        """
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        a = self.a_embedder(a)  # (N, D) Action embedding
        t = self.t_embedder(t)  # (N, D)
        c = t + a
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


class DiTAction(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=8,
        patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        action_dim=6,  # Added action dimension
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else 3  # in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.a_embedder = ActionEmbedder(
            hidden_size, action_dim
        )  # Added action embedder
        self.img_c_embedder = nn.Linear(256, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # TODO: Add current image as embedding https://github.com/homangab/Track-2-Act/blob/main/single_script.py
    def forward(self, x, t, a, img_c):
        """
        Forward pass of DiT which now also takes actions as input.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        a: (N, 6) tensor of actions (TCP positions in (x, y, z, rpx, rpy, rpz))
        """
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        a = self.a_embedder(a)  # (N, D) Action embedding
        t = self.t_embedder(t)  # (N, D)
        y_feat = img_c.flatten(start_dim=1)
        y_emb = self.img_c_embedder(y_feat)
        c = t + a + y_emb
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


class DiTActionSeq(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=16,
        patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        action_dim=6,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else 4
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.a_embedder = ActionEmbedder(hidden_size, action_dim)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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


class DiTActionSeqAct(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=16,
        patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        action_dim=6,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else 4
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.a_embedder = ActionEmbedder(hidden_size, action_dim)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode="linear"
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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


class DiTActionFramesSeq(DiTActionSeqAct):
    def __init__(
        self,
        input_size=16,
        patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        action_dim=6,
        learn_sigma=True,
        seq_l=16,
    ):
        super().__init__(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            action_dim,
            learn_sigma,
        )
        self.seq_length = seq_l
        self.pos_embed_act = nn.Parameter(
            torch.zeros(1, 16, hidden_size), requires_grad=False
        )

        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_length, hidden_size), requires_grad=False
        )

        temp_embed = get_1d_sincos_temp_embed(
            self.temp_embed.shape[-1], self.temp_embed.shape[-2]
        )
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        pos_embed_act = get_2d_sincos_pos_embed(
            self.pos_embed_act.shape[-1], int(self.seq_length**0.5)
        )
        self.pos_embed_act.data.copy_(
            torch.from_numpy(pos_embed_act).float().unsqueeze(0)
        )
        self.img_c_embedder = nn.Linear(256, hidden_size)
        self.final_layer_act = FinalLayer(hidden_size, self.seq_length + 1, action_dim)
        self.downsample_layer = nn.Linear(2023, 7)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer_act.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_act.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_act.linear.weight, 0)
        nn.init.constant_(self.final_layer_act.linear.bias, 0)
        nn.init.normal_(self.downsample_layer.weight, std=0.02)

    def forward(self, x, t, a, img_c, mask_frame_num=None):
        """
        Forward pass of DiT which now also takes actions as input
        x: (N, L, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        a: (N, C, 7) tensor of actions (TCP positions in (x, y, z, rpx, rpy, rpz))
        """
        batch_sz, l, ch, h, w = x.shape
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed

        a = self.a_embedder(a) + self.pos_embed_act  # (N, D) Action embedding
        t = self.t_embedder(t)  # (N, D)
        a = rearrange(a, "b f d -> (b f) d")
        a = repeat(a, "b d -> b c d", c=1)
        x = torch.cat((x, a), dim=1)

        # timestep_spatial = repeat(t, "n d -> (n c) d", c=l)
        # y_feat = img_c.flatten(start_dim=1)
        # y_emb = self.img_c_embedder(y_feat)
        # c = timestep_spatial + y_emb
        timestep_spatial = repeat(t, "n d -> (n c) d", c=l)
        timestep_temp = repeat(t, "n d -> (n c) d", c=(self.pos_embed.shape[1] + 1))
        y_feat = img_c.flatten(start_dim=1)
        y_emb = self.img_c_embedder(y_feat)

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]
            c = timestep_spatial + y_emb
            x = spatial_block(x, c)
            x = rearrange(x, "(b f) t d -> (b t) f d", b=batch_sz)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed[:, 0:l]
            c = timestep_temp
            x = temp_block(x, c)
            x = rearrange(x, "(b t) f d -> (b f) t d", b=batch_sz)
        x_b = x
        # for block in self.blocks:
        #     x_b = block(x, c)  # (N, T, D)
        #      x_b[:, :-1, :], c
        # )  # (N, T, patch_size ** 2 * out_channels)
        # x_act = self.final_layer_act(x_b[:, 15:16, :], c)
        # x_act = torch.einsum("nhw->nhw", x_act)
        # x_act = x_act.view((batch_sz, l, -1))  # torch.Size([32, 16, 7])
        # x_act = self.downsample_layer(x_act)
        # x = self.unpatchify(x)  # (N, out_channels, H, W)
        # x = rearrange(x, "(b f) c h w -> b f c h w", b=batch_sz)
        # x = self.final_layer(
        #     x_b[:, :-1, :], c
        # )  # (N, T, patch_size ** 2 * out_channels)
        c = timestep_spatial + y_emb
        x_act = self.final_layer_act(x_b[:, 15:16, :], c)
        x_act = torch.einsum("nhw->nhw", x_act)
        x_act = x_act.view((batch_sz, l, -1))  # torch.Size([32, 16, 7])
        x_act = self.downsample_layer(x_act)
        x = self.final_layer(x_b[:, :-1, :], c)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=batch_sz)
        return x, x_act


class DiTActionSeqISim(DiTActionSeq):
    # https://github.com/bytedance/IRASim/blob/main/models/irasim.py
    def __init__(
        self,
        input_size=16,
        patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        action_dim=6,
        learn_sigma=True,
        seq_len=45,
    ):
        super().__init__(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            action_dim,
            learn_sigma,
        )
        self.temp_embed = nn.Parameter(
            torch.zeros(1, seq_len, hidden_size), requires_grad=False
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


class DiTActionFramesSeq2(DiTActionSeqAct):
    def __init__(
        self,
        input_size=16,
        patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        action_dim=6,
        learn_sigma=True,
        seq_l=16,
        mask_n=2,
    ):
        super().__init__(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            action_dim,
            learn_sigma,
        )
        self.seq_length = seq_l
        self.mask_n = mask_n

        self.a_embedder = ActionEmbedder(hidden_size * 8, action_dim)

        self.pos_embed_act = nn.Parameter(
            torch.zeros(1, self.mask_n, hidden_size * 8), requires_grad=False
        )

        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_length, hidden_size), requires_grad=False
        )

        self.last_block = DiTBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode="math"
        )

        nn.init.constant_(self.last_block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.last_block.adaLN_modulation[-1].bias, 0)

        temp_embed = get_1d_sincos_temp_embed(
            self.temp_embed.shape[-1], self.temp_embed.shape[-2]
        )
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        pos_embed_act = get_2d_sincos_pos_embed(
            self.pos_embed_act.shape[-1], int(self.mask_n**0.5)
        )
        self.pos_embed_act.data.copy_(
            torch.from_numpy(pos_embed_act).float().unsqueeze(0)
        )
        self.img_c_embedder = nn.Linear(256, hidden_size)
        self.final_layer_act = FinalLayer(hidden_size, self.seq_length + 1, action_dim)
        self.downsample_layer = nn.Linear(2023, 7)

        nn.init.normal_(self.a_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.a_embedder.mlp[2].weight, std=0.02)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer_act.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_act.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_act.linear.weight, 0)
        nn.init.constant_(self.final_layer_act.linear.bias, 0)
        nn.init.normal_(self.downsample_layer.weight, std=0.02)

    def forward(self, x, t, a, img_c, mask_frame_num=None):
        """
        Forward pass of DiT which now also takes actions as input
        x: (N, L, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        a: (N, C, 7) tensor of actions (TCP positions in (x, y, z, rpx, rpy, rpz))
        """
        batch_sz, l, ch, h, w = x.shape
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed
        a = self.a_embedder(a) + self.pos_embed_act  # (N, D) Action embedding
        t = self.t_embedder(t)  # (N, D)
        # Change to masked token
        a = rearrange(a, "b f d -> (b f) d")
        a = rearrange(a, "b (c h) -> (b c) h", c=8)
        a = repeat(a, "b d -> b c d", c=1)
        # a_ext = torch.zeros_like(x).to(x.device)
        # b, f, h = a.shape
        # a_ext[:b, :f, :h] = a
        # a = rearrange(a, "b c d -> (b c) d")
        # a = repeat(a, "b d -> b c d", c=1)
        x = torch.cat((x, a), dim=1)
        # x = x + a
        timestep_spatial = repeat(t, "n d -> (n c) d", c=l)
        y_feat = img_c.flatten(start_dim=1)
        y_emb = self.img_c_embedder(y_feat)
        c = timestep_spatial + y_emb
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x_b = self.last_block(x, c)
        x_act = self.final_layer_act(x_b[:, 15:16, :], c)
        x_act = torch.einsum("nhw->nhw", x_act)
        x_act = x_act.view((batch_sz, l, -1))  # torch.Size([32, 16, 7])
        x_act = self.downsample_layer(x_act)

        x = self.final_layer(x_b[:, :-1, :], c)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=batch_sz)
        return x, x_act


class DiTActionFramesSeq3(DiTActionSeqAct):
    def __init__(
        self,
        input_size=16,
        patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        action_dim=6,
        learn_sigma=True,
        seq_l=16,
        mask_n=2,
    ):
        super().__init__(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            action_dim,
            learn_sigma,
        )
        self.seq_length = seq_l
        self.mask_n = mask_n

        self.a_embedder = ActionEmbedder(hidden_size * 5, action_dim)

        self.pos_embed_act = nn.Parameter(
            torch.zeros(1, self.mask_n, hidden_size * 5), requires_grad=False
        )

        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_length, hidden_size), requires_grad=False
        )

        self.last_block = DiTBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode="math"
        )

        nn.init.constant_(self.last_block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.last_block.adaLN_modulation[-1].bias, 0)

        temp_embed = get_1d_sincos_temp_embed(
            self.temp_embed.shape[-1], self.temp_embed.shape[-2]
        )
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        pos_embed_act = get_2d_sincos_pos_embed(
            self.pos_embed_act.shape[-1], int(self.mask_n**0.5)
        )
        self.pos_embed_act.data.copy_(
            torch.from_numpy(pos_embed_act).float().unsqueeze(0)
        )
        self.img_c_embedder = nn.Linear(256, hidden_size)
        self.final_layer_act = FinalLayer(hidden_size, self.seq_length + 1, action_dim)
        self.downsample_layer = nn.Linear(847, 7)

        nn.init.normal_(self.a_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.a_embedder.mlp[2].weight, std=0.02)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer_act.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_act.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_act.linear.weight, 0)
        nn.init.constant_(self.final_layer_act.linear.bias, 0)
        nn.init.normal_(self.downsample_layer.weight, std=0.02)

    def forward(self, x, t, a, img_c, mask_frame_num=None):
        """
        Forward pass of DiT which now also takes actions as input
        x: (N, L, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        a: (N, C, 7) tensor of actions (TCP positions in (x, y, z, rpx, rpy, rpz))
        """
        batch_sz, l, ch, h, w = x.shape
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed
        a = self.a_embedder(a) + self.pos_embed_act  # (N, D) Action embedding
        t = self.t_embedder(t)  # (N, D)
        # Change to masked token
        a = rearrange(a, "b f d -> (b f) d")
        a = rearrange(a, "b (c h) -> (b c) h", c=5)
        a = repeat(a, "b d -> b c d", c=1)
        x = torch.cat((x, a), dim=1)
        # x = x + a
        timestep_spatial = repeat(t, "n d -> (n c) d", c=l)
        timestep_temp = repeat(t, "n d -> (n c) d", c=(self.pos_embed.shape[1] + 1))

        y_feat = img_c.flatten(start_dim=1)
        y_emb = self.img_c_embedder(y_feat)

        for i in range(0, len(self.blocks), 2):
            c = timestep_spatial + y_emb
            spatial_block, temp_block = self.blocks[i : i + 2]
            x = spatial_block(x, c)
            x = rearrange(x, "(b f) t d -> (b t) f d", b=batch_sz)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed[:, 0:l]
            c = timestep_temp
            x = temp_block(x, c)
            x = rearrange(x, "(b t) f d -> (b f) t d", b=batch_sz)

        c = timestep_spatial + y_emb
        x = self.last_block(x, c)
        x_act = self.final_layer_act(x[:, 15:16, :], c)
        x_act = torch.einsum("nhw->nhw", x_act)
        x_act = x_act.view((batch_sz, l, -1))  # torch.Size([32, 16, 7])
        x_act = self.downsample_layer(x_act)
        x = self.final_layer(x[:, :-1, :], c)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=batch_sz)
        return x, x_act


class DiTActionFramesSeq4(DiTActionSeqAct):
    def __init__(
        self,
        input_size=16,
        patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        action_dim=6,
        learn_sigma=True,
        seq_l=16,
        mask_n=2,
    ):
        super().__init__(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            action_dim,
            learn_sigma,
        )
        self.seq_length = seq_l
        self.mask_n = mask_n

        self.a_embedder = ActionEmbedder(hidden_size * 5, action_dim)

        self.pos_embed_act = nn.Parameter(
            torch.zeros(1, self.mask_n, hidden_size * 5), requires_grad=False
        )

        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_length, hidden_size), requires_grad=False
        )

        # Replace DiTBlock with JointTransformerBlock
        self.first_block = DiTBlockJoint(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode="math"
        )

        temp_embed = get_1d_sincos_temp_embed(
            self.temp_embed.shape[-1], self.temp_embed.shape[-2]
        )
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        pos_embed_act = get_2d_sincos_pos_embed(
            self.pos_embed_act.shape[-1], int(self.mask_n**0.5)
        )
        self.pos_embed_act.data.copy_(
            torch.from_numpy(pos_embed_act).float().unsqueeze(0)
        )
        self.img_c_embedder = nn.Linear(256, hidden_size)
        self.final_layer_act = FinalLayer(hidden_size, self.seq_length + 1, action_dim)
        self.downsample_layer = nn.Linear(847, 7)

        nn.init.constant_(self.first_block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.first_block.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.a_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.a_embedder.mlp[2].weight, std=0.02)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer_act.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_act.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_act.linear.weight, 0)
        nn.init.constant_(self.final_layer_act.linear.bias, 0)
        nn.init.normal_(self.downsample_layer.weight, std=0.02)

    def forward(self, x, t, a, img_c, mask_frame_num=None):
        """
        Forward pass of DiT which now also takes actions as input
        x: (N, L, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        a: (N, C, 7) tensor of actions (TCP positions in (x, y, z, rpx, rpy, rpz))
        """
        batch_sz, l, ch, h, w = x.shape
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed
        a = self.a_embedder(a) + self.pos_embed_act  # (N, D) Action embedding
        t = self.t_embedder(t)  # (N, D)
        # Change to masked token
        a = rearrange(a, "b f d -> (b f) d")
        a = rearrange(a, "b (c h) -> (b c) h", c=5)
        a = repeat(a, "b d -> b c d", c=1)
        # x = torch.cat((x, a), dim=1)
        # x = x + a
        timestep_spatial = repeat(t, "n d -> (n c) d", c=l)
        timestep_temp = repeat(t, "n d -> (n c) d", c=(self.pos_embed.shape[1] + 1))

        y_feat = img_c.flatten(start_dim=1)
        y_emb = self.img_c_embedder(y_feat)
        c = timestep_spatial + y_emb
        x = self.first_block(x, a, c)

        for i in range(0, len(self.blocks), 2):
            c = timestep_spatial + y_emb
            spatial_block, temp_block = self.blocks[i : i + 2]
            x = spatial_block(x, c)
            x = rearrange(x, "(b f) t d -> (b t) f d", b=batch_sz)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed[:, 0:l]
            c = timestep_temp
            x = temp_block(x, c)
            x = rearrange(x, "(b t) f d -> (b f) t d", b=batch_sz)

        c = timestep_spatial + y_emb

        x_act = self.final_layer_act(x[:, 15:16, :], c)
        x_act = torch.einsum("nhw->nhw", x_act)
        x_act = x_act.view((batch_sz, l, -1))  # torch.Size([32, 16, 7])
        x_act = self.downsample_layer(x_act)
        x = self.final_layer(x[:, :-1, :], c)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=batch_sz)
        return x, x_act


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
    def __init__(
        self,
        input_size=16,
        patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        action_dim=6,
        learn_sigma=True,
        seq_len=45,
    ):
        super().__init__(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            action_dim,
            learn_sigma,
        )
        self.temp_embed = nn.Parameter(
            torch.zeros(1, seq_len, hidden_size), requires_grad=False
        )
        temp_embed = get_1d_sincos_temp_embed(
            self.temp_embed.shape[-1], self.temp_embed.shape[-2]
        )
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))
        self.condition_emb = ConditionEmbedding(out_dim=96)

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
        c_m = rearrange(c_m, "b l c w h-> (b l) c w h")
        canny_emb = self.condition_emb(c_m)
        canny_emb = rearrange(canny_emb, "b c w h -> b (c w h)")
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


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
