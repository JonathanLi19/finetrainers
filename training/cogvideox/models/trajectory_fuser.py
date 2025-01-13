import torch
from torch import nn
from einops import rearrange

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class FloatGroupNorm(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.to(self.bias.dtype)).type(x.dtype)
    
class TrajectoryFuser(nn.Module):
    def __init__(self, in_channel, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.gamma_spatial = nn.Conv2d(in_channel, self.out_channels // 4, 3, padding=1)
        self.gamma_temporal = zero_module(
            nn.Conv1d(
                self.out_channels // 4,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            )
        )
        self.beta_spatial = nn.Conv2d(in_channel, self.out_channels // 4, 3, padding=1)
        self.beta_temporal = zero_module(
            nn.Conv1d(
                self.out_channels // 4,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            )
        )
        self.flow_cond_norm = FloatGroupNorm(32, self.out_channels)
        self.scale = 1.0

    def forward(self, x, trajectory_maps, T):
        gamma = self.gamma_spatial(trajectory_maps)
        beta = self.beta_spatial(trajectory_maps)
        _, _, hh, wh = beta.shape
        gamma = rearrange(gamma, "(B T) C H W -> (B H W) C T", T=T)
        beta = rearrange(beta, "(B T) C H W -> (B H W) C T", T=T)
        gamma = self.gamma_temporal(gamma)
        beta = self.beta_temporal(beta)
        gamma = rearrange(gamma, "(B H W) C T -> (B T) C H W", H=hh, W=wh)
        beta = rearrange(beta, "(B H W) C T -> (B T) C H W", H=hh, W=wh)
        print(gamma.mean(), beta.mean())
        x = x + (self.flow_cond_norm(x) * gamma + beta) * self.scale
        return x