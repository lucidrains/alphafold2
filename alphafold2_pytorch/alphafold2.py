import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import torch.nn.functional as F

from einops import rearrange, repeat

# constants

MAX_NUM_MSA = 20
NUM_AMINO_ACIDS = 21
NUM_EMBEDDS_TR  = 1280 # best esm model 
DISTOGRAM_BUCKETS = 37

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, mask = None, context_mask = None):
        device, h, has_context = x.device, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class AxialAttention(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.attn_width = Attention(**kwargs)
        self.attn_height = Attention(**kwargs)

    def forward(self, x, context = None, mask = None, context_mask = None):
        b, h, w, d = x.shape
        w_mask = h_mask = w_context = h_context = w_context_mask = h_context_mask = None

        if exists(mask):
            w_mask = rearrange(mask, 'b h w -> (b w) h', w = w)
            h_mask = rearrange(mask, 'b h w -> (b h) w', h = h)

        if exists(context):
            w_context = repeat(context, 'b n d -> (b w) n d', w = w)
            h_context = repeat(context, 'b n d -> (b h) n d', h = h)
            w_context_mask = repeat(context_mask, 'b n -> (b w) n', w = w)
            h_context_mask = repeat(context_mask, 'b n -> (b h) n', h = h)

        w_x = rearrange(x, 'b h w d -> (b w) h d')
        w_out = self.attn_width(w_x, mask = w_mask, context = w_context, context_mask = w_context_mask)
        w_out = rearrange(w_out, '(b w) h d -> b h w d', h = h, w = w)

        h_x = rearrange(x, 'b h w d -> (b h) w d')
        h_out = self.attn_height(h_x, mask = h_mask, context = h_context, context_mask = h_context_mask)
        h_out = rearrange(h_out, '(b h) w d -> b h w d', h = h, w = w)

        return w_out + h_out

# main class

class Alphafold2(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_seq_len = 2048,
        depth = 6,
        heads = 8,
        dim_head = 64,
        pos_token = 3,
        num_tokens = NUM_AMINO_ACIDS,
        num_embedds = NUM_EMBEDDS_TR,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # multiple sequence alignment position embedding

        self.msa_pos_emb = nn.Embedding(max_seq_len, dim)
        self.msa_num_pos_emb = nn.Embedding(MAX_NUM_MSA, dim)

        # custom embedding projection

        self.embedd_project = nn.Linear(num_embedds, dim)

        wrapper = partial(PreNorm, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                wrapper(AxialAttention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                wrapper(Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                wrapper(FeedForward(dim = dim, dropout = ff_dropout)),
            ]))


        self.msa_layers = nn.ModuleList([])
        for _ in range(depth):
            self.msa_layers.append(nn.ModuleList([
                wrapper(Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                wrapper(Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                wrapper(FeedForward(dim = dim, dropout = ff_dropout))
            ]))

        self.norm = nn.LayerNorm(dim)
        self.to_distogram_logits = nn.Linear(dim, DISTOGRAM_BUCKETS)

    def forward(self, seq, msa = None, embedds = None, mask = None, msa_mask = None):
        n, device = seq.shape[1], seq.device

        # unpack (AA_cod, atom_pos)
        if isinstance(seq, list):
            seq, seq_pos = seq
        # embed main sequence
        x = self.token_emb(seq)
        x += self.pos_emb(torch.arange(n, device = device))[None, ...]
        x = x[:, :, None, :] + x[:, None, :, :] # create pair-wise residue embeds
        x_mask = mask[:, :, None] * mask[:, None, :]

        # embed multiple sequence alignment

        if exists(msa):
            msa_shape = msa.shape

            m = self.token_emb(msa)
            m += self.msa_pos_emb(torch.arange(msa_shape[-1], device = device))[None, None, ...]
            m += self.msa_num_pos_emb(torch.arange(msa_shape[1], device = device))[None, :, None, :]

            m = rearrange(m, 'b m n d -> b (m n) d')

        elif exists(embedds):
            m = self.embedd_project(embedds)
            # pairwise mat - maybe repeating stuff? 
            m = m[:, :, None, :] + m[:, None, :, :]
            m = rearrange(m, 'b m n d -> b (m n) d')

        if exists(msa_mask):
            msa_mask = rearrange(msa_mask, 'b m n -> b (m n)')

        # trunk

        for ((attn, cross_attn, ff), (msa_attn, msa_cross_attn, msa_ff)) in zip(self.layers, self.msa_layers):

            # self attention

            x = attn(x, mask = x_mask) + x

            if exists(msa):
                m = msa_attn(m, mask = msa_mask) + m

                # cross attention

                x = rearrange(x, 'b i j d -> b (i j) d')
                x_mask_flat = rearrange(x_mask, 'b i j -> b (i j)')

                m = msa_cross_attn(
                    m,
                    mask = msa_mask,
                    context = x,
                    context_mask = x_mask_flat
                ) + m

                x = cross_attn(
                    x,
                    mask = x_mask_flat,
                    context = m,
                    context_mask = msa_mask
                ) + x

                x = rearrange(x, 'b (i j) d -> b i j d', i = n)

            # feedforwards

            x = ff(x) + x

            if exists(msa):
                m = msa_ff(m) + m

        x = self.norm(x)

        x = (x + rearrange(x, 'b i j d -> b j i d')) * 0.5  # symmetrize
        return self.to_distogram_logits(x)
