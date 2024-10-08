import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from einops import rearrange, repeat
from .difussion_utils.transformers_utils import (
    DiTBlock,
    DiTBlockJoint,
    DiTBlockJoint2,
    MMDiTBlockJoint,
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


class DiTActionSeqAct(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, config):
        super(DiTActionSeqAct, self).__init__()
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
                DiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attention_mode=config["attention_mode"],
                )
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

    def __init__(self, config):
        super().__init__(config)
        self.pos_embed_act = nn.Parameter(
            torch.zeros(1, 16, self.hidden_size), requires_grad=False
        )

        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.hidden_size), requires_grad=False
        )

        temp_embed = get_1d_sincos_temp_embed(
            self.temp_embed.shape[-1], self.temp_embed.shape[-2]
        )
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        pos_embed_act = get_2d_sincos_pos_embed(
            self.pos_embed_act.shape[-1], int(self.seq_len**0.5)
        )
        self.pos_embed_act.data.copy_(
            torch.from_numpy(pos_embed_act).float().unsqueeze(0)
        )
        self.img_c_embedder = nn.Linear(256, self.hidden_size)
        self.final_layer_act = FinalLayer(
            self.hidden_size, self.seq_len + 1, self.action_dim
        )
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
        c = timestep_spatial + y_emb
        x_act = self.final_layer_act(x_b[:, 15:16, :], c)
        x_act = torch.einsum("nhw->nhw", x_act)
        x_act = x_act.view((batch_sz, l, -1))  # torch.Size([32, 16, 7])
        x_act = self.downsample_layer(x_act)
        x = self.final_layer(x_b[:, :-1, :], c)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=batch_sz)
        return x, x_act


class DiTActionFramesSeq2(DiTActionSeqAct):
    def __init__(self, config):
        super().__init__(config)
        self.a_embedder = ActionEmbedder(self.hidden_size * 8, self.action_dim)

        self.pos_embed_act = nn.Parameter(
            torch.zeros(1, self.mask_n, self.hidden_size * 8), requires_grad=False
        )

        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.hidden_size), requires_grad=False
        )

        self.last_block = DiTBlock(
            self.hidden_size,
            self.num_heads,
            mlp_ratio=self.mlp_ratio,
            attention_mode="math",
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
        self.img_c_embedder = nn.Linear(256, self.hidden_size)
        self.final_layer_act = FinalLayer(
            self.hidden_size, self.seq_len + 1, self.action_dim
        )
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

    def __init__(self, config):
        super().__init__(config)

        self.a_embedder = ActionEmbedder(self.hidden_size * 5, self.action_dim)

        self.pos_embed_act = nn.Parameter(
            torch.zeros(1, self.mask_n, self.hidden_size * 5), requires_grad=False
        )

        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.hidden_size), requires_grad=False
        )

        self.last_block = DiTBlock(
            self.hidden_size,
            self.num_heads,
            mlp_ratio=self.mlp_ratio,
            attention_mode="math",
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
        self.img_c_embedder = nn.Linear(256, self.hidden_size)
        self.final_layer_act = FinalLayer(
            self.hidden_size, self.seq_len + 1, self.action_dim
        )
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

    def __init__(self, config):
        super().__init__(config)

        self.a_embedder = ActionEmbedder(self.hidden_size * 5, self.action_dim)

        self.pos_embed_act = nn.Parameter(
            torch.zeros(1, self.mask_n, self.hidden_size * 5), requires_grad=False
        )

        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.hidden_size), requires_grad=False
        )

        # Replace DiTBlock with JointTransformerBlock
        self.first_block = DiTBlockJoint(
            self.hidden_size,
            self.num_heads,
            mlp_ratio=self.mlp_ratio,
            attention_mode="math",
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
        self.img_c_embedder = nn.Linear(256, self.hidden_size)
        self.final_layer_act = FinalLayer(
            self.hidden_size, self.seq_len + 1, self.action_dim
        )
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


class DiTActionFramesSeq5(DiTActionSeqAct):

    def __init__(self, config):
        super().__init__(config)

        self.a_embedder = ActionEmbedder(self.hidden_size * 5, self.action_dim)

        self.pos_embed_act = nn.Parameter(
            torch.zeros(1, self.mask_n, self.hidden_size * 5), requires_grad=False
        )

        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.hidden_size), requires_grad=False
        )

        # Replace DiTBlock with JointTransformerBlock
        self.first_block = DiTBlockJoint2(
            self.hidden_size,
            self.num_heads,
            mlp_ratio=self.mlp_ratio,
            attention_mode="math",
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
        self.img_c_embedder = nn.Linear(256, self.hidden_size)
        self.final_layer_act = FinalLayer(
            self.hidden_size, self.seq_len + 1, self.action_dim
        )
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


class DiTActionFramesSeq6(DiTActionSeqAct):

    def __init__(self, config):
        super().__init__(config)

        self.a_embedder = ActionEmbedder(self.hidden_size * 5, self.action_dim)

        self.pos_embed_act = nn.Parameter(
            torch.zeros(1, self.mask_n, self.hidden_size * 5), requires_grad=False
        )

        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.hidden_size), requires_grad=False
        )

        self.first_block = MMDiTBlockJoint(
            self.hidden_size,
            self.num_heads,
            mlp_ratio=self.mlp_ratio,
            attention_mode="math",
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
        self.img_c_embedder = nn.Linear(256, self.hidden_size)
        self.final_layer_act = FinalLayer(
            self.hidden_size, self.seq_len + 1, self.action_dim
        )
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
