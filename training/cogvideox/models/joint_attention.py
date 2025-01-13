import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
import os
from sklearn.decomposition import PCA

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
        
        self.is_causal = False

    def forward(self, x: torch.Tensor, T=None, H=None, W=None, depth=None, first_frame=False,) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # print(q.shape, k.shape, v.shape) # Spatial: [15, 16, 1590, 72]; Temporal: [1590, 16, 15, 72]
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
        # print(q.shape, k.shape, v.shape) # Spatial: [15, 16, 1590, 72]; Temporal: [1590, 16, 15, 72]
        if enable_flash_attn:
            from flash_attn import flash_attn_func

            if first_frame:
                k = rearrange(k, "(B T) H S D -> B T H S D", T=T)
                k = k[:, [0] * T]
                k = rearrange(k, "B T H S D -> (B T) H S D")

                v = rearrange(v, "(B T) H S D -> B T H S D", T=T)
                v = v[:, [0] * T]
                v = rearrange(v, "B T H S D -> (B T) H S D")

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            # print(q.shape, k.shape, v.shape) # [15, 1590, 16, 72]
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )
            # print(x.shape) # [15, 1590, 16, 72]
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
                attn += causal_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
    
class JointAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias, qk_norm, rope, enable_flash_attn, scale=1):
        super().__init__()

        self.attn = Attention(hidden_size,
                            num_heads=num_heads,
                            qkv_bias=qkv_bias,
                            qk_norm=qk_norm,
                            rope=rope,
                            enable_flash_attn=enable_flash_attn,)
        self.ff = FeedForward(hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor([0.])) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor([0.])) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = scale


    def forward(self, x, objs):
        N_visual = x.shape[1]
        # print(self.alpha_attn)
        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x,objs],dim=1)))[:,0:N_visual,:]
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))  
        
        return x   