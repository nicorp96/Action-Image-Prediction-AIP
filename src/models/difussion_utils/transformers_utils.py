import torch.nn as nn
from torch import einsum
import torch
import torch.nn.functional as F

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
            self.bucket_size = 64 if not self.bucket_size is None else self.bucket_size
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
