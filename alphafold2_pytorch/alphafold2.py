import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat, reduce

import alphafold2_pytorch.constants as constants
from alphafold2_pytorch.reversible import ReversibleSequence

from se3_transformer_pytorch import SE3Transformer

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

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
        seq_len = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        compress_ratio = 1,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

        # memory compressed attention
        self.compress_ratio = compress_ratio
        self.compress_fn = nn.Conv1d(inner_dim, inner_dim, compress_ratio, stride = compress_ratio, groups = heads) if compress_ratio > 1 else None

    def forward(self, x, context = None, mask = None, context_mask = None):
        device, h, has_context = x.device, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        if exists(self.compress_fn):
            assert has_context, 'memory compressed attention only works in the context of cross attention for now'

            ratio = self.compress_ratio
            padding = ratio - (j % ratio)

            if padding < ratio:
                k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0), (k ,v))

                if exists(context_mask):
                    context_mask = F.pad(context_mask, (0, padding), value = False)

                k, v = map(lambda t: rearrange(t, 'b n c -> b c n'), (k, v))
                k, v = map(self.compress_fn, (k, v))
                k, v = map(lambda t: rearrange(t, 'b c n -> b n c'), (k, v))

                if exists(context_mask):
                    context_mask = reduce(context_mask.float(), 'b (n r) -> b n', 'sum', r = ratio)
                    context_mask = context_mask > 0

                j = (j + padding) // ratio

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(i, device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(j, device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SparseAttention(Attention):
    def __init__(
        self,
        *args,
        block_size = 16,
        num_random_blocks = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert exists(self.seq_len), '`seq_len` must be defined if using sparse attention class'
        from deepspeed.ops.sparse_attention import SparseSelfAttention, VariableSparsityConfig
        self.block_size = block_size
        num_random_blocks = default(num_random_blocks, self.seq_len // block_size // 4)

        self.attn_fn = SparseSelfAttention(
            sparsity_config = VariableSparsityConfig(
                num_heads = self.heads,
                block = self.block_size,
                num_random_blocks = num_random_blocks,
                attention = 'bidirectional'
            ),
            max_seq_length = self.seq_len,
            attn_mask_mode = 'add'
        )

    def forward(self, x, mask = None):
        device, h = x.device, self.heads
        n = x.shape[1]

        remainder = x.shape[1] % self.block_size

        if remainder > 0:
            padding = self.block_size - remainder
            x = F.pad(x, (0, 0, 0, padding), value = 0)
            mask = F.pad(mask, (0, padding), value = False)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        key_pad_mask = None
        if exists(mask):
            key_pad_mask = ~mask

        out = self.attn_fn(q, k, v, key_padding_mask = key_pad_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out[:, :n]

class AxialAttention(nn.Module):
    def __init__(
        self,
        sparse_attn = False,
        **kwargs
    ):
        super().__init__()
        attn_class = SparseAttention if sparse_attn else Attention
        self.attn_width = attn_class(**kwargs)
        self.attn_height = attn_class(**kwargs)

    def forward(self, x, context = None, mask = None, context_mask = None):
        b, n, d = x.shape
        w = h = int(sqrt(n))

        x = x.reshape(b, h, w, d)
        mask = mask.reshape(b, h, w) if exists(mask) else None

        w_mask = h_mask = w_context = h_context = w_context_mask = h_context_mask = None

        if exists(mask):
            w_mask = rearrange(mask, 'b h w -> (b w) h', w = w)
            h_mask = rearrange(mask, 'b h w -> (b h) w', h = h)

        if exists(context):
            w_context = repeat(context, 'b n d -> (b w) n d', w = w)
            h_context = repeat(context, 'b n d -> (b h) n d', h = h)
            w_context_mask = repeat(context_mask, 'b n -> (b w) n', w = w)
            h_context_mask = repeat(context_mask, 'b n -> (b h) n', h = h)

        attn_kwargs = {} if not exists(context) else {'context': w_context, 'context_mask': w_context_mask}
        w_x = rearrange(x, 'b h w d -> (b w) h d')
        w_out = self.attn_width(w_x, mask = w_mask, **attn_kwargs)
        w_out = rearrange(w_out, '(b w) h d -> b h w d', h = h, w = w)

        h_x = rearrange(x, 'b h w d -> (b h) w d')
        h_out = self.attn_height(h_x, mask = h_mask, **attn_kwargs)
        h_out = rearrange(h_out, '(b h) w d -> b h w d', h = h, w = w)

        out = w_out + h_out
        return rearrange(out, 'b h w d -> b (h w) d')

# main class

class SequentialSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, m, mask = None,  msa_mask = None, **kwargs):
        for ((attn, ff, msa_attn), (cross_attn, msa_ff, msa_cross_attn)) in zip(*[iter(self.blocks)] * 2):

            # self attention

            x = attn(x, mask = mask) + x

            if exists(m):
                m = msa_attn(m, mask = msa_mask) + m

                # cross attention

                x = cross_attn(x, m, mask = mask, context_mask = msa_mask) + x
                m = msa_cross_attn(m, x, mask = msa_mask, context_mask = mask) + m

            # feedforwards

            x = ff(x) + x

            if exists(m):
                m = msa_ff(m) + m

        return x, m

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
        reversible = False,
        sparse_self_attn = False,
        cross_attn_compress_ratio = 1,
    ):
        super().__init__()
        layers_sparse_attn = cast_tuple(sparse_self_attn, depth)

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.pos_emb_ax = nn.Embedding(max_seq_len, dim)

        # multiple sequence alignment position embedding

        self.msa_pos_emb = nn.Embedding(max_seq_len, dim)
        self.msa_num_pos_emb = nn.Embedding(constants.MAX_NUM_MSA, dim)

        # custom embedding projection

        self.embedd_project = nn.Linear(num_embedds, dim)

        # main trunk modules

        prenorm = partial(PreNorm, dim)
        prenorm_cross = partial(PreNormCross, dim)

        layers = nn.ModuleList([])
        for _, layer_sparse_attn in zip(range(depth), layers_sparse_attn):

            # self attention, for both main sequence and msa

            attn_class = SparseAttention if layer_sparse_attn else Attention

            layers.append(nn.ModuleList([
                prenorm(AxialAttention(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, sparse_attn = sparse_self_attn)),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout)),
                prenorm(attn_class(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout)),
            ]))

            # cross attention, for main sequence -> msa and then msa -> sequence

            layers.append(nn.ModuleList([
                prenorm_cross(Attention(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, compress_ratio = cross_attn_compress_ratio)),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout)),
                prenorm_cross(Attention(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, compress_ratio = cross_attn_compress_ratio)),
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

        # use axial positional embedding

        seq_range = torch.arange(n, device = device)
        ax1 = x + self.pos_emb(seq_range)[None, ...]
        ax2 = x + self.pos_emb_ax(seq_range)[None, ...]

        # outer sum

        x = rearrange(ax1, 'b i d -> b i () d') + rearrange(ax2, 'b j d-> b () j d') # create pair-wise residue embeds
        x_mask = rearrange(mask, 'b i -> b i ()') + rearrange(mask, 'b j -> b () j') if exists(mask) else None

        x = rearrange(x, 'b i j d -> b (i j) d')
        x_mask = rearrange(x_mask, 'b i j -> b (i j)') if exists(mask) else None

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
            m = rearrange(m, 'b i d -> b i () d') + rearrange(m, 'b j d -> b () j d')
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
