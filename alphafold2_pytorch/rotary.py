from math import log, sqrt, pi
import torch
from torch import nn, einsum

from einops import rearrange, repeat

# rotary embedding helpers

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(x, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    return x * cos + rotate_every_two(x) * sin

# positional embeddings

class DepthWiseConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True, groups = None):
        super().__init__()
        groups = default(groups, dim_in)
        self.net = nn.Sequential(
            nn.Conv1d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = groups, stride = stride, bias = bias),
            nn.Conv1d(dim_in, dim_out, 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, n, device):
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinusoid_inp = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return rearrange(emb, 'i j -> () i j')

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim // 4
        inv_freq = 1. / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, n, device):
        seq = torch.arange(n, device = device).type_as(self.inv_freq)

        x = einsum('n, d -> n d', seq, self.inv_freq)
        y = einsum('n, d -> n d', seq, self.inv_freq)

        x_sinu = repeat(x, 'i d -> i j d', j = n)
        y_sinu = repeat(y, 'j d -> i j d', i = n)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos
