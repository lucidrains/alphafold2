import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat, reduce

import alphafold2_pytorch.constants as constants
from alphafold2_pytorch.utils import get_bucketed_distance_matrix, center_distogram_torch, MDScaling
from alphafold2_pytorch.reversible import ReversibleSequence

from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.utils import torch_default_dtype

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
        context = self.norm_context(context)
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
        tie_attn_dim = None
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

        self.tie_attn_dim = tie_attn_dim

    def forward(self, x, context = None, mask = None, context_mask = None, tie_attn_dim = None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        # memory compressed attention, to make cross-attention more efficient

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

        # for tying row-attention, for MSA axial self-attention

        if exists(tie_attn_dim):
            q, k, v = map(lambda t: rearrange(t, '(b r) h n d -> b r h n d', r = tie_attn_dim), (q, k, v))

            # when tying row-attention, one cannot have any masked out tokens
            if exists(mask):
                assert torch.all(mask), 'you cannot have any padding if you are to tie the row attention across MSAs'
                mask = None

            dots = einsum('b r h i d, b r h j d -> b h i j', q, k) * self.scale * (tie_attn_dim ** -0.5)
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # masking

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(1, j, device = device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots.masked_fill_(~mask, mask_value)

        # attention

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        if exists(tie_attn_dim):
            out = einsum('b h i j, b r h j d -> b r h i d', attn, v)
            out = rearrange(out, 'b r h n d -> (b r) h n d')
        else:
            out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # combine heads and project out

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class SparseAttention(Attention):
    def __init__(
        self,
        *args,
        block_size = 16,
        num_random_blocks = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert not exists(self.tie_attn_dim), 'sparse attention is not compatible with tying of row attention'
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
        device, orig_shape, h = x.device, x.shape, self.heads

        b, n, _ = x.shape
        assert n <= self.seq_len, f'either the AA sequence length {n} or the total MSA length {n} exceeds the allowable sequence length {self.seq_len} for sparse attention, set by `max_seq_len`'

        remainder = x.shape[1] % self.block_size

        if remainder > 0:
            padding = self.block_size - remainder
            x = F.pad(x, (0, 0, 0, padding), value = 0)
            mask = torch.ones(b, n, device = device).bool()
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
        out = out[:, :n]

        return out

class AxialAttention(nn.Module):
    def __init__(
        self,
        tie_row_attn = False,
        sparse_attn = False,
        **kwargs
    ):
        super().__init__()
        attn_class = SparseAttention if sparse_attn else Attention

        self.tie_row_attn = tie_row_attn # tie the row attention, from the paper 'MSA Transformer'

        self.attn_width = attn_class(**kwargs)
        self.attn_height = attn_class(**kwargs)

    def forward(self, x, shape, context = None, mask = None, context_mask = None):
        b, n, d = x.shape

        x = x.view(shape)
        h, w = shape[1:3]

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

        tie_attn_dim = x.shape[1] if self.tie_row_attn else None
        h_x = rearrange(x, 'b h w d -> (b h) w d')
        h_out = self.attn_height(h_x, mask = h_mask, tie_attn_dim = tie_attn_dim, **attn_kwargs)
        h_out = rearrange(h_out, '(b h) w d -> b h w d', h = h, w = w)

        out = w_out + h_out
        return rearrange(out, 'b h w d -> b (h w) d')

# main class

class SequentialSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(
        self,
        x,
        m,
        seq_shape,
        msa_shape,
        mask = None,
        msa_mask = None,
        **kwargs
    ):
        for ((attn, ff, msa_attn), (cross_attn, msa_ff, msa_cross_attn)) in zip(*[iter(self.blocks)] * 2):

            # self attention

            x = attn(x, seq_shape, mask = mask) + x

            if exists(m):
                m = msa_attn(m, msa_shape, mask = msa_mask) + m

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
        num_tokens = constants.NUM_AMINO_ACIDS,
        num_embedds = constants.NUM_EMBEDDS_TR,
        max_num_msas = constants.MAX_NUM_MSA,
        max_num_templates = constants.MAX_NUM_TEMPLATES,
        attn_dropout = 0.,
        ff_dropout = 0.,
        reversible = False,
        sparse_self_attn = False,
        cross_attn_compress_ratio = 1,
        msa_tie_row_attn = False,
        template_attn_depth = 2,
        predict_coords = False, # structure module related keyword arguments below
        mds_iters = 2,
        structure_module_dim = 4,
        structure_module_depth = 4,
        structure_module_heads = 1,
        structure_module_dim_head = 16,
        structure_module_refinement_iters = 2,
        structure_module_knn = 8
    ):
        super().__init__()
        layers_sparse_attn = cast_tuple(sparse_self_attn, depth)

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.pos_emb_ax = nn.Embedding(max_seq_len, dim)

        # multiple sequence alignment position embedding

        self.msa_pos_emb = nn.Embedding(max_seq_len, dim)
        self.msa_num_pos_emb = nn.Embedding(max_num_msas, dim)

        # template embedding

        self.template_dist_emb = nn.Embedding(constants.DISTOGRAM_BUCKETS, dim)
        self.template_num_pos_emb = nn.Embedding(max_num_templates, dim)
        self.template_pos_emb = nn.Embedding(max_seq_len, dim)
        self.template_pos_emb_ax = nn.Embedding(max_seq_len, dim)

        # template sidechain encoding

        self.sidechains_proj = nn.Parameter(torch.randn(1, dim))

        self.template_sidechain_emb = SE3Transformer(
            dim = dim,
            dim_head = dim,
            heads = 1,
            num_neighbors = 12,
            depth = 4,
            input_degrees = 2,
            num_degrees = 2,
            output_degrees = 1,
            reversible = True
        )

        # custom embedding projection

        self.embedd_project = nn.Linear(num_embedds, dim)

        # main trunk modules

        prenorm = partial(PreNorm, dim)
        prenorm_cross = partial(PreNormCross, dim)

        template_layers = nn.ModuleList([])
        for _ in range(template_attn_depth):
            template_layers.append(nn.ModuleList([
                prenorm(AxialAttention(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                prenorm(AxialAttention(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                prenorm(Attention(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout))
            ]))

        self.template_attn_net = template_layers

        layers = nn.ModuleList([])
        for _, layer_sparse_attn in zip(range(depth), layers_sparse_attn):

            # self attention, for both main sequence and msa

            layers.append(nn.ModuleList([
                prenorm(AxialAttention(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, sparse_attn = sparse_self_attn)),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout)),
                prenorm(AxialAttention(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, tie_row_attn = msa_tie_row_attn)),
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

        # to distogram output

        self.to_distogram_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, constants.DISTOGRAM_BUCKETS)
        )

        # to coordinate output

        self.predict_coords = predict_coords
        self.mds_iters = mds_iters
        self.structure_module_refinement_iters = structure_module_refinement_iters

        with torch_default_dtype(torch.float64):
            self.structure_module_embeds = nn.Embedding(num_tokens, structure_module_dim)
            self.to_refined_coords_delta = nn.Linear(structure_module_dim, 1)

            self.structure_module = SE3Transformer(
                dim = structure_module_dim,
                depth = structure_module_depth,
                input_degrees = 1,
                num_degrees = 3,
                output_degrees = 2,
                heads = structure_module_heads,
                num_neighbors = structure_module_knn,
                differentiable_coors = True
            )

    def forward(
        self,
        seq,
        msa = None,
        mask = None,
        msa_mask = None,
        templates_seq = None,
        templates_dist = None,
        templates_mask = None,
        templates_coors = None,
        templates_sidechains = None,
        embedds = None,
    ):
        n, device = seq.shape[1], seq.device
        n_range = torch.arange(n, device = device)

        # unpack (AA_code, atom_pos)

        if isinstance(seq, (list, tuple)):
            seq, seq_pos = seq

        # embed main sequence

        x = self.token_emb(seq)

        # outer sum

        x = rearrange(x, 'b i d -> b i () d') + rearrange(x, 'b j d-> b () j d') # create pair-wise residue embeds
        x_mask = rearrange(mask, 'b i -> b i ()') + rearrange(mask, 'b j -> b () j') if exists(mask) else None

        # flatten

        seq_shape = x.shape
        x = rearrange(x, 'b i j d -> b (i j) d')
        x_mask = rearrange(x_mask, 'b i j -> b (i j)') if exists(mask) else None

        # axial positional embedding

        pos_emb = rearrange(self.pos_emb(n_range), 'i d -> () i () d') + rearrange(self.pos_emb_ax(n_range), 'j d -> () () j d')
        x += rearrange(pos_emb, 'b i j d -> b (i j) d')

        # embed multiple sequence alignment (msa)

        m = None
        msa_shape = None
        if exists(msa):
            m = self.token_emb(msa)
            m += self.msa_pos_emb(torch.arange(msa.shape[-1], device = device))[None, None, ...]
            m += self.msa_num_pos_emb(torch.arange(msa.shape[1], device = device))[None, :, None, :]

            msa_shape = m.shape
            m = rearrange(m, 'b m n d -> b (m n) d')

        elif exists(embedds):
            m = self.embedd_project(embedds)
            m = rearrange(m, 'b i d -> b i () d') + rearrange(m, 'b j d -> b () j d')
            m = rearrange(m, 'b m n d -> b (m n) d')

        if exists(msa_mask):
            msa_mask = rearrange(msa_mask, 'b m n -> b (m n)')

        # trunk (template attention, wip)

        if exists(templates_seq):
            assert exists(templates_coors), 'template residue coordinates must be supplied `templates_coors`'
            _, num_templates, *_ = templates_seq.shape


            if not exists(templates_dist):
                templates_dist = get_bucketed_distance_matrix(templates_coors, templates_mask, constants.DISTOGRAM_BUCKETS)

            # embed template

            t_seq = self.token_emb(templates_seq)

            # if sidechain information is present
            # color the residue embeddings with the sidechain type 1 features
            # todo (make efficient)

            if exists(templates_sidechains):
                shape = t_seq.shape

                t_seq = rearrange(t_seq, 'b t n d-> (b t) n d ()')

                templates_sidechains = rearrange(templates_sidechains, 'b t n m -> (b t) n () m')
                templates_sidechains = einsum('b n d m, d e -> b n e m', templates_sidechains, self.sidechains_proj)
                templates_coors = rearrange(templates_coors, 'b t n m -> (b t) n m')

                se3_templates_mask = rearrange(templates_mask, 'b t n -> (b t) n')

                t_seq = self.template_sidechain_emb(
                    {'0': t_seq, '1': templates_sidechains},
                    templates_coors,
                    mask = se3_templates_mask,
                    return_type = 0
                )

                t_seq = t_seq.reshape(*shape)

            # embed template distances

            t_dist = self.template_dist_emb(templates_dist)

            t_seq = rearrange(t_seq, 'b t i d -> b t i () d') + rearrange(t_seq, 'b t j d -> b t () j d')
            t = t_seq + t_dist

            template_shape = rearrange(t, 'b t h w d -> (b t) h w d').shape

            # template pos emb

            template_num_pos_emb = self.template_num_pos_emb(torch.arange(num_templates, device = device))

            t += rearrange(template_num_pos_emb, 't d-> () t () () d')
            t = rearrange(t, 'b t h w d -> (b t) (h w) d')

            pos_emb = rearrange(self.template_pos_emb(n_range), 'i d -> () i () d') + rearrange(self.template_pos_emb_ax(n_range), 'j d -> () () j d')
            pos_emb = rearrange(pos_emb, 'b i j d -> b (i j) d')
            t += pos_emb

            if exists(templates_mask):
                t_mask = rearrange(templates_mask, 'b t i -> b t i ()') * rearrange(templates_mask, 'b t j -> b t () j')
                t_mask = rearrange(t_mask, 'b t h w -> (b t) (h w)')

            assert t.shape[-2:] == x.shape[-2:]

            # template self attention followed by cross attention axially to primary sequence

            for (seq_self_attn, template_self_attn, cross_attn, ff) in self.template_attn_net:
                x = seq_self_attn(x, mask = x_mask, shape = seq_shape)
                t = template_self_attn(t, mask = t_mask, shape = template_shape) + t

                # concat template and primary sequence, for axial attention along the template dimension
                # very similar to the attention across the time axis from https://arxiv.org/abs/2102.05095

                x = rearrange(x, 'b n d -> (b n) () d', n = n ** 2)
                t = rearrange(t, '(b t) n d -> (b n) t d', t = num_templates)

                y_mask = None
                if exists(templates_mask) and exists(x_mask):
                    t_mask_ = rearrange(t_mask, '(b t) n -> (b n) t', t = num_templates)
                    mask_ = rearrange(x_mask, 'b n -> (b n) ()')
                    y_mask = torch.cat((mask_, t_mask_), dim = 1)

                y = torch.cat((x, t), dim = 1)
                y = cross_attn(y, mask = y_mask) + y

                x = rearrange(y[:, 0], '(b n) d -> b n d', n = n ** 2)
                t = rearrange(y[:, 1:], '(b n) t d -> (b t) n d', n = (n ** 2))

                t = ff(t) + t

        # trunk

        x, m = self.net(
            x,
            m,
            seq_shape,
            msa_shape,
            mask = x_mask,
            msa_mask = msa_mask
        )

        x = rearrange(x, 'b (h w) d -> b h w d', h = n)
        x = (x + rearrange(x, 'b i j d -> b j i d')) * 0.5  # symmetrize
        distogram_logits = self.to_distogram_logits(x)

        if not self.predict_coords:
            return distogram_logits

        # structural refinement

        coords = []
        for distogram in distogram_logits:
            distances, weights = center_distogram_torch(distogram)

            coords_3d, _ = MDScaling(distances, 
                weights = weights,
                iters = 5, 
                fix_mirror = 0
            )

            coords_3d = rearrange(coords_3d, 'b c n -> b n c')
            coords.append(coords_3d)

        coords = torch.cat(coords, dim = 0)
        x = self.structure_module_embeds(seq)

        original_dtype = coords.dtype

        x, coords = map(lambda t: t.double(), (x, coords))

        with torch_default_dtype(torch.float64):
            for _ in range(self.structure_module_refinement_iters):
                output = self.structure_module(x, coords, mask = mask)
                x, refined_coords = output['0'], output['1']

                refined_coords = rearrange(refined_coords, 'b n d c -> b n c d')
                refined_coords = self.to_refined_coords_delta(refined_coords)
                refined_coords = rearrange(refined_coords, 'b n c () -> b n c')

                x = rearrange(x, 'b n c () -> b n c')
                coords = coords + refined_coords

        coords.type(original_dtype)
        return coords
