import torch.nn as nn
from torch import einsum
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Mlp
import numpy as np

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except:
    pass

try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        attention_mode="math",
        **block_kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.eps = 1e-3
        self.bucket_size = 4
        self.kv_mask = None
        if block_kwargs is not None:
            for k in block_kwargs.keys():
                if k == "kv_mask":
                    self.kv_mask = block_kwargs["kv_mask"]
                if k == "bucket_size":
                    self.bucket_size = block_kwargs["bucket_size"]
                if k == "eps":
                    self.eps = block_kwargs["eps"]

    def forward(self, x):
        B, N, C = x.shape

        if self.attention_mode == "xformers":  # cause loss nan while using with amp
            x = xformers.ops.memory_efficient_attention(q, k, v).reshape(B, N, C)

        elif self.attention_mode == "flash":
            # Flash Attention
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .contiguous()
            )
            attn_output = flash_attn_qkvpacked_func(qkv, causal=False, dropout_p=0.1)
            # attn_output = attn_output.permute(0, 2, 1, 3) # (b,nh, l, c) -> (b, l, nh, c
            x = attn_output.reshape(B, N, C)

        elif self.attention_mode == "math":
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
                .contiguous()
            )
            q, k, v = qkv.unbind(
                0
            )  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        elif self.attention_mode == "linear":
            # https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
                .contiguous()
            )
            q, k, v = qkv.unbind(0)
            dim = q.shape[-1]

            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)

            q = q * dim**-0.5

            context = einsum("bhnd,bhne->bhde", k, v)
            attn = einsum("bhnd,bhde->bhne", q, context)
            attn = attn.reshape(*q.shape)
            x = attn.transpose(1, 2).reshape(B, N, C)

        elif self.attention_mode == "causal_linear":
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
                .contiguous()
            )
            q, k, v = qkv.unbind(0)

            b, h, n, e, dtype = *q.shape, q.dtype
            self.bucket_size = 4 if not self.bucket_size is None else self.bucket_size
            self.bucket_size = max(self.bucket_size, 1)
            assert (
                self.bucket_size == 0 or (n % self.bucket_size) == 0
            ), f"sequence length {n} must be divisible by the bucket size {self.bucket_size} for causal linear attention"

            q = q.softmax(dim=-1)
            k = torch.exp(k).type(dtype).clone()

            q = q * e**-0.5

            if self.kv_mask is not None:
                mask = self.kv_mask[:, None, :, None]
                k = k.masked_fill_(~mask, 0.0)
                v = v.masked_fill_(~mask, 0.0)
                del mask

            bucket_fn = lambda x: x.reshape(*x.shape[:-2], -1, self.bucket_size, e)
            b_q, b_k, b_v = map(bucket_fn, (q, k, v))

            b_k_sum = b_k.sum(dim=-2)
            b_k_cumsum = b_k_sum.cumsum(dim=-2).type(dtype)

            context = einsum("bhund,bhune->bhude", b_k, b_v)
            context = context.cumsum(dim=-3).type(dtype)

            if self.bucket_size > 1:
                context = F.pad(context, (0, 0, 0, 0, 1, 0), value=0.0)
                context, _ = split_at_index(2, -1, context)

                b_k_cumsum = F.pad(b_k_cumsum, (0, 0, 1, 0), value=0.0)
                b_k_cumsum, _ = split_at_index(2, -1, b_k_cumsum)

            D_inv = 1.0 / einsum("bhud,bhund->bhun", b_k_cumsum, b_q).clamp(
                min=self.eps
            )
            attn = einsum("bhund,bhude,bhun->bhune", b_q, context, D_inv)
            x = attn.transpose(1, 2).reshape(B, N, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
        **block_kwargs,
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


class DiTBlockJoint(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attention_mode="math",
        **block_kwargs,
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

    def forward(self, x, a, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        a = a + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(a), shift_msa, scale_msa)
        )
        a = a + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(a), shift_mlp, scale_mlp)
        )
        out = torch.cat((x, a), dim=1)
        return out


class DiTBlockJoint2(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attention_mode="math",
        **block_kwargs,
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
        self.attn2 = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attention_mode=attention_mode,
            **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        self.mlp2 = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, a, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        a = a + gate_msa.unsqueeze(1) * self.attn2(
            modulate(self.norm3(a), shift_msa, scale_msa)
        )
        a = a + gate_mlp.unsqueeze(1) * self.mlp2(
            modulate(self.norm4(a), shift_mlp, scale_mlp)
        )
        out = torch.cat((x, a), dim=1)
        return out


class MMDiTBlockJoint(nn.Module):
    """
    Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
    https://arxiv.org/pdf/2403.03206
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attention_mode="math",
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.lin1 = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.lin2 = nn.Linear(hidden_size, 3 * hidden_size, bias=True)

        self.attn1 = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, device="cuda:0"
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.lin3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.lin4 = nn.Linear(hidden_size, hidden_size, bias=True)

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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

        self.adaLN_modulation_x = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, a, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = (
            self.adaLN_modulation_x(c).chunk(6, dim=1)
        )
        a_q, a_k, a_v = self.lin1(modulate(self.norm1(a), shift_msa, scale_msa)).chunk(
            3, dim=2
        )
        x_q, x_k, x_v = self.lin2(
            modulate(self.norm2(x), shift_msa_x, scale_msa_x)
        ).chunk(3, dim=2)
        x = torch.cat((x, a), dim=1)
        query = torch.cat((x_q, a_q), dim=1)
        key = torch.cat((x_k, a_k), dim=1)
        value = torch.cat((x_v, a_v), dim=1)
        x_c, _ = self.attn1(query, key, value)
        a_att = gate_msa.unsqueeze(1) * self.lin3(x_c) + a
        a_out = (
            gate_mlp.unsqueeze(1)
            * self.mlp(modulate(self.norm3(a_att), shift_mlp, scale_mlp))
            + a
        )
        x_att = gate_msa_x.unsqueeze(1) * self.lin4(x_c) + x
        x_out = (
            gate_mlp_x.unsqueeze(1)
            * self.mlp(modulate(self.norm4(x_att), shift_mlp_x, scale_mlp_x))
            + x
        )
        x = x_out + a_out
        return x


class DiTBlockFrameAttention(nn.Module):
    """
    Text-to-Image Diffusion Models are Zero-Shot Video Generators
    https://arxiv.org/pdf/2303.13439
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        num_msk=2,
        **block_kwargs,
    ):
        super().__init__()
        self.num_msk = num_msk
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.lin1 = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.lin2 = nn.Linear(hidden_size, hidden_size, bias=True)
        head_dim = hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation_x = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        B, N, C = x.shape
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = (
            self.adaLN_modulation_x(c).chunk(6, dim=1)
        )
        x_0 = x[:, : self.num_msk, :]
        x_0_k, x_0_v = self.lin1(
            modulate(self.norm1(x_0), shift_msa_x, scale_msa_x)
        ).chunk(2, dim=2)

        x_k_q = self.lin2(modulate(self.norm2(x), shift_msa_x, scale_msa_x))

        attn = (x_k_q @ x_0_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ x_0_v).transpose(1, 2).reshape(B, N, C)

        x = x + gate_msa_x.unsqueeze(1) * x

        x = x + gate_mlp_x.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp_x, scale_mlp_x)
        )

        return x
