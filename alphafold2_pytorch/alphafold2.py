import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat

import alphafold2_pytorch.constants as constants
from alphafold2_pytorch.reversible import ReversibleSequence, SequentialSequence

from se3_transformer_pytorch import SE3Transformer

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
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class PreNormCross(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

    def forward(self, x, context, *args, **kwargs):
        x = self.norm(x)
        context = self.norm(context)
        return self.fn(x, context, *args, **kwargs)

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
        b, n, d = x.shape
        w = h = int(sqrt(n))

        x = x.reshape(b, h, w, d)
        mask = mask.reshape(b, h, w)

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

        out = w_out + h_out
        return rearrange(out, 'b h w d -> b (h w) d')

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
        num_tokens = constants.NUM_AMINO_ACIDS,
        num_embedds = constants.NUM_EMBEDDS_TR,
        attn_dropout = 0.,
        ff_dropout = 0.,
        reversible = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # multiple sequence alignment position embedding

        self.msa_pos_emb = nn.Embedding(max_seq_len, dim)
        self.msa_num_pos_emb = nn.Embedding(constants.MAX_NUM_MSA, dim)

        # custom embedding projection

        self.embedd_project = nn.Linear(num_embedds, dim)

        # main trunk modules

        prenorm = partial(PreNorm, dim)
        prenorm_cross = partial(PreNormCross, dim)

        layers = nn.ModuleList([])
        for _ in range(depth):

            # self attention, for both main sequence and msa

            layers.append(nn.ModuleList([
                prenorm(AxialAttention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout)),
                prenorm(Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout)),
            ]))

            # cross attention, for main sequence -> msa and then msa -> sequence

            layers.append(nn.ModuleList([
                prenorm_cross(Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout)),
                prenorm_cross(Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout)),
            ]))

        if not reversible:
            layers = nn.ModuleList(list(map(lambda t: t[:3], layers))) # remove last feed forward if not reversible

        trunk_class = SequentialSequence if not reversible else ReversibleSequence
        self.net = trunk_class(layers)

        # to output

        self.to_distogram_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, constants.DISTOGRAM_BUCKETS)
        )

    def forward(self, seq, msa = None, embedds = None, mask = None, msa_mask = None):
        n, device = seq.shape[1], seq.device

        # unpack (AA_code, atom_pos)

        if isinstance(seq, (list, tuple)):
            seq, seq_pos = seq

        # embed main sequence

        x = self.token_emb(seq)
        x += self.pos_emb(torch.arange(n, device = device))[None, ...]
        x = x[:, :, None, :] + x[:, None, :, :] # create pair-wise residue embeds
        x_mask = mask[:, :, None] * mask[:, None, :]

        x = rearrange(x, 'b i j d -> b (i j) d')
        x_mask = rearrange(x_mask, 'b i j -> b (i j)')

        # embed multiple sequence alignment

        m = None
        if exists(msa):
            msa_shape = msa.shape

            m = self.token_emb(msa)
            m += self.msa_pos_emb(torch.arange(msa_shape[-1], device = device))[None, None, ...]
            m += self.msa_num_pos_emb(torch.arange(msa_shape[1], device = device))[None, :, None, :]

            m = rearrange(m, 'b m n d -> b (m n) d')

        elif exists(embedds):
            m = self.embedd_project(embedds)
            m = m[:, :, None, :] + m[:, None, :, :]
            m = rearrange(m, 'b m n d -> b (m n) d')

        if exists(msa_mask):
            msa_mask = rearrange(msa_mask, 'b m n -> b (m n)')

        # trunk

        x, m = self.net(x, m, mask = x_mask, msa_mask = msa_mask)

        # structural refinement

        ### TODO - use SE3Transformer here, as details emerge about the iterative refinement, fill-in here

        # final out, do alphafold1's distogram for now

        x = rearrange(x, 'b (h w) d -> b h w d', h = n)
        x = (x + rearrange(x, 'b i j d -> b j i d')) * 0.5  # symmetrize
        return self.to_distogram_logits(x)
