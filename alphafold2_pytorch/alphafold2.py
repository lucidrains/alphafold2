import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
from itertools import islice, cycle
from collections import namedtuple
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

import alphafold2_pytorch.constants as constants
from alphafold2_pytorch.utils import *
from alphafold2_pytorch.reversible import ReversibleSequence
from alphafold2_pytorch.rotary import DepthWiseConv1d, AxialRotaryEmbedding, FixedPositionalEmbedding, apply_rotary_pos_emb

# structure module

from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.utils import torch_default_dtype, fourier_encode
from en_transformer import EnTransformer
from egnn_pytorch import EGNN_Network

from performer_pytorch import FastAttention, ProjectionUpdater

# constants

Logits = namedtuple('Logits', ['distance', 'theta', 'phi', 'omega'])

# helpers

def exists(val):
    return val is not None

def maybe(fn):
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None
        return fn(t, *args, **kwargs)
    return inner

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# helper classes

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return self.val

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

# interceptor functions, for slicing and splicing of template tensors, which have been concatted to the primary sequence

class InterceptFeedForward(nn.Module):
    def __init__(
        self,
        slice_tuple = None,
        ff = None
    ):
        super().__init__()
        assert exists(ff), 'feedforward not given'
        self.ff = ff
        self.slice_tuple = slice_tuple

    def forward(self, x, *args, shape = None, mask = None, **kwargs):
        assert exists(shape), 'sequence shape must be given for intercepting and slicing of inputs'
        ff, slice_tuple = self.ff, self.slice_tuple

        if not exists(slice_tuple):
            return ff(x, *args, **kwargs)

        x = x.view(shape)
        output = torch.zeros_like(x)
        x = x[slice_tuple]

        output_subset_shape = x.shape
        x = rearrange(x, 'b ... d -> b (...) d')

        ff_output = self.ff(x, *args, **kwargs)

        ff_output = ff_output.view(output_subset_shape)
        output[slice_tuple] = ff_output
        return rearrange(output, 'b ... d -> b (...) d')

class InterceptAxialAttention(nn.Module):
    def __init__(
        self,
        slice_tuple = None,
        attn = None
    ):
        super().__init__()
        assert exists(attn), 'attention function not given'
        self.attn = attn
        self.slice_tuple = slice_tuple

    def forward(self, x, *args, shape = None, mask = None, **kwargs):
        assert exists(shape), 'sequence shape must be given for intercepting and slicing of inputs'
        attn, slice_tuple = self.attn, self.slice_tuple

        if not exists(slice_tuple):
            return attn(x, *args, shape = shape, mask = mask, **kwargs)

        x = x.view(shape)
        output = torch.zeros_like(x)
        x = x[slice_tuple]
        output_subset_shape = x.shape
        x = rearrange(x, 'b ... d -> b (...) d')

        if exists(mask):
            mask = mask.view(shape[:-1])
            mask = mask[slice_tuple]
            mask = rearrange(mask, 'b ... -> b (...)')

        attn_output = attn(x, *args, shape = output_subset_shape, mask = mask, **kwargs)

        attn_output = attn_output.view(output_subset_shape)
        output[slice_tuple] = attn_output
        return rearrange(output, 'b ... d -> b (...) d')

class InterceptAttention(nn.Module):
    def __init__(
        self,
        slice_tuple,
        attn = None,
        context = False,
    ):
        super().__init__()
        assert exists(attn), 'attention function not given'
        self.attn = attn
        self.slice_tuple = slice_tuple
        self.context = context

    def forward(self, *args, shape = None, context_shape = None, mask = None, context_mask = None, **kwargs):
        assert exists(shape) and exists(context_shape), 'sequence and context shape must be given for intercepting and slicing of inputs'
        attn, slice_tuple, context = self.attn, self.slice_tuple, self.context

        x, c, *args = args

        if context:
            c = c.view(context_shape)
            c = c[slice_tuple]
            c = rearrange(c, 'b ... d -> b (...) d')

            if exists(context_mask):
                context_mask = context_mask.view(context_shape[:-1])
                context_mask = context_mask[slice_tuple]
                context_mask = rearrange(context_mask, 'b ... -> b (...)')
        else:
            x = x.view(shape)
            output = torch.zeros_like(x)
            x = x[slice_tuple]
            output_subset_shape = x.shape
            x = rearrange(x, 'b ... d -> b (...) d')

            if exists(mask):
                mask = mask.view(shape[:-1])
                mask = mask[slice_tuple]
                mask = rearrange(mask, 'b ... -> b (...)')

        attn_output = attn(x, c, *args, shape = shape, context_shape = context_shape, mask = mask, context_mask = context_mask, **kwargs)

        if context:
            return attn_output

        attn_output = attn_output.view(output_subset_shape)
        output[slice_tuple] = attn_output
        return rearrange(output, 'b ... d -> b (...) d')

# kronecker attention wrapper

def norm_shape(shape): # hack to squeeze out extra dimension for templates
    if len(shape) == 5:
        return torch.Size([shape[0], *shape[2:]])
    return shape

def kron_operator(t, shape, mask = None, fn = torch.sum):
    if exists(mask):
        t = t.masked_fill_(~mask[..., None], 0.)

    t = t.reshape(*shape)
    t = torch.cat((fn(t, dim = 2), fn(t, dim = 1)), dim = 1)

    if exists(mask):
        mask = mask.reshape(*shape[:-1])
        mask = torch.cat((mask.any(dim = 2), mask.any(dim = 1)), dim = 1)

    return t, mask

class KronInputWrapper(nn.Module):
    def __init__(
        self,
        fn,
        kron_queries = False,
        kron_context = False
    ):
        super().__init__()
        self.fn = fn
        self.kron_queries = kron_queries
        self.kron_context = kron_context

    def forward(self, x, context, *args, shape = None, context_shape = None, mask = None, context_mask = None, rotary_emb = None, **kwargs):
        assert not (self.kron_queries and not exists(shape)), 'shape for input must be given if queries are to be kroneckered'
        assert not (self.kron_context and not exists(context_shape)), 'shape for context must be given if context are to be kroneckered'

        shape, context_shape = map(norm_shape, (shape, context_shape))

        if self.kron_queries or self.kron_context:
            rotary_emb = None # turn off rotary embeddings if kron is being used, for now

        if self.kron_context:
            context, context_mask = kron_operator(context, context_shape, context_mask)

        if self.kron_queries:
            x, mask = kron_operator(x, shape, mask)

        out = self.fn(x, context, *args, mask = mask, context_mask = context_mask, rotary_emb = rotary_emb, **kwargs)

        if self.kron_queries:
            out_h, out_w = out.split(shape[1:3], dim = 1)
            out = rearrange(out_h, 'b h d -> b h () d') + rearrange(out_w, 'b w d -> b () w d')
            out = rearrange(out, 'b ... d -> b (...) d')

        return out

# feed forward

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True, groups = None):
        super().__init__()
        groups = default(groups, dim_in)
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = groups, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

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

    def forward(self, x, **kwargs):
        return self.net(x)

class LocalFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout = 0.,
        kernel_size = 3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            DepthWiseConv2d(hidden_dim, hidden_dim, kernel_size, padding = kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1)
        )
    def forward(self, x):
        h = w = int(sqrt(x.shape[-2]))
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        x = self.net(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

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

    def forward(self, x, context = None, mask = None, context_mask = None, tie_attn_dim = None, rotary_emb = None, **kwargs):
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

        # rotary relative positional encoding

        if exists(rotary_emb):
            rot_q, rot_k = cast_tuple(rotary_emb, 2)
            q = apply_rotary_pos_emb(q, rot_q)
            k = apply_rotary_pos_emb(k, rot_k)

        # for tying row-attention, for MSA axial self-attention

        if exists(tie_attn_dim):
            q, k, v = map(lambda t: rearrange(t, '(b r) h n d -> b r h n d', r = tie_attn_dim), (q, k, v))

            if exists(mask):
                mask = rearrange(mask, '(b r) n -> b r n', r = tie_attn_dim)
                has_rows = mask.any(dim = -1)

                num_rows = has_rows.sum(dim = -1)
                num_rows = rearrange(num_rows, 'b -> b () () ()').to(q)
                mask = mask.any(dim = 1)

                # mask out the rows that have nothing as 0
                row_mask = ~rearrange(has_rows, 'b r -> b r () () ()')
                q.masked_fill_(row_mask, 0.)
            else:
                num_rows = tie_attn_dim

            dots = einsum('b r h i d, b r h j d -> b h i j', q, k) * self.scale * (num_rows ** -0.5)
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

class LinearAttention(Attention):
    def __init__(self, *args, dim_head = 64, nb_features = 256, **kwargs):
        kwargs.update(dim_head = dim_head)
        super().__init__(*args, **kwargs)

        self.fast_attn = FastAttention(
            dim_heads = dim_head,
            nb_features = nb_features
        )

    def forward(self, x, context = None, mask = None, context_mask = None, rotary_emb = None, **kwargs):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # rotary relative positional encoding

        if exists(rotary_emb):
            rot_q, rot_k = cast_tuple(rotary_emb, 2)
            q = apply_rotary_pos_emb(q, rot_q)
            k = apply_rotary_pos_emb(k, rot_k)
            v = apply_rotary_pos_emb(v, rot_k)

        # linear attention masking

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(1, j, device = device).bool())
            v = v.masked_fill(context_mask[:, None, :, None], 0.)

        # fast linear attention

        out = self.fast_attn(q, k, v)

        # combine heads and project out

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

    def forward(self, x, mask = None, rotary_emb = None, **kwargs):
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

        if exists(rotary_emb):
            rot_q, rot_k = cast_tuple(rotary_emb, 2)
            q = apply_rotary_pos_emb(q, rot_q)
            k = apply_rotary_pos_emb(k, rot_k)
            v = apply_rotary_pos_emb(v, rot_k)

        key_pad_mask = None
        if exists(mask):
            key_pad_mask = repeat(~mask, 'b n -> b h n', h = h)

        out = self.attn_fn(q, k, v, key_padding_mask = key_pad_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out[:, :n]

        return out

class AxialAttention(nn.Module):
    def __init__(
        self,
        template_axial_attn = False,
        tie_row_attn = False,
        sparse_attn = False,
        row_attn = True,
        col_attn = True,
        **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'

        attn_class = SparseAttention if sparse_attn else Attention

        self.tie_row_attn = tie_row_attn # tie the row attention, from the paper 'MSA Transformer'

        self.attn_width = attn_class(**kwargs)
        self.attn_height = attn_class(**kwargs)

        self.template_axial_attn = template_axial_attn
        if template_axial_attn:
            self.attn_frames = Attention(**kwargs)

        self.row_attn = row_attn
        self.col_attn = col_attn

    def forward(self, x, shape, mask = None, rotary_emb = None):
        b, n, d = x.shape

        num_axials = len(shape) - 2
        x = x.view(shape)

        h, w = shape[-3:-1]

        # axial mask

        if num_axials == 3:
            t = shape[1]
        else:
            t = 1
            x = rearrange(x, 'b h w d -> b () h w d')
            mask = rearrange(mask, 'b n -> b () n')

        w_mask = h_mask = None

        if exists(mask):
            mask = mask.reshape(b, t, h, w)
            w_mask = rearrange(mask, 'b t h w -> (b t w) h')
            h_mask = rearrange(mask, 'b t h w -> (b t h) w')

            if num_axials == 3:
                f_mask = rearrange(mask, 'b t h w -> (b h w) t')

        # axial pos emb

        h_rotary_emb, w_rotary_emb = cast_tuple(rotary_emb, 2)

        # axial attention

        out = 0
        axial_attn_count = 0

        if self.row_attn:
            w_x = rearrange(x, 'b t h w d -> (b t w) h d')
            w_out = self.attn_width(w_x, mask = w_mask, rotary_emb = w_rotary_emb)
            w_out = rearrange(w_out, '(b t w) h d -> b t h w d', h = h, w = w, t = t)

            out += w_out
            axial_attn_count += 1

        if self.col_attn:
            tie_attn_dim = x.shape[2] if self.tie_row_attn else None
            h_x = rearrange(x, 'b t h w d -> (b t h) w d')
            h_out = self.attn_height(h_x, mask = h_mask, tie_attn_dim = tie_attn_dim, rotary_emb = h_rotary_emb)
            h_out = rearrange(h_out, '(b t h) w d -> b t h w d', h = h, w = w, t = t)

            out += h_out
            axial_attn_count += 1

        # do attention across the templates dimension, if (1) templates are present and (2) template axial attention was activated for the module

        needs_template_axial_attn = x.shape[1] > 1 and self.template_axial_attn
        if needs_template_axial_attn:
            f_x = rearrange(x, 'b t h w d -> (b h w) t d')
            f_out = self.attn_frames(f_x, mask = f_mask)
            f_out = rearrange(f_out, '(b h w) t d -> b t h w d', h = h, w = w, t = t)

            out += f_out
            axial_attn_count += 1

        out /= axial_attn_count

        return rearrange(out, 'b t h w d -> b (t h w) d')

# template module helpers and classes

class SE3TemplateEmbedder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = SE3Transformer(*args, **kwargs)
        self.sidechains_proj = nn.Parameter(torch.randn(1, kwargs['dim']))

    def forward(
        self,
        t_seq,
        templates_sidechains,
        templates_coors,
        mask = None
    ):
        shape = t_seq.shape
        t_seq = rearrange(t_seq, 'b t n d-> (b t) n d ()')

        templates_sidechains = rearrange(templates_sidechains, 'b t n m -> (b t) n () m')
        templates_sidechains = einsum('b n d m, d e -> b n e m', templates_sidechains, self.sidechains_proj)
        templates_coors = rearrange(templates_coors, 'b t n m -> (b t) n m')

        mask = rearrange(mask, 'b t n -> (b t) n')

        t_seq = self.net(
            {'0': t_seq, '1': templates_sidechains},
            templates_coors,
            mask = mask,
            return_type = 0
        )

        t_seq = t_seq.reshape(*shape)
        return t_seq

# structure module helpers and classes

class SE3TransformerWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = SE3Transformer(*args, **kwargs)
        self.to_refined_coords_delta = nn.Linear(kwargs['dim'], 1)

    def forward(self, x, coords, mask = None, adj_mat = None, edges = None, global_feats = None):
        output = self.net(x, coords, mask = mask, adj_mat = adj_mat, edges = edges, global_feats = global_feats)
        x, refined_coords = output['0'], output['1']

        refined_coords = rearrange(refined_coords, 'b n d c -> b n c d')
        refined_coords = self.to_refined_coords_delta(refined_coords)
        refined_coords = rearrange(refined_coords, 'b n c () -> b n c')

        coords = coords + refined_coords
        return x, coords

# coupling module, from trunk to initial coordinates for refinement

class CoordModuleMDS(nn.Module):
    def __init__(
        self,
        mds_iters,
        use_eigen_mds,
        predict_real_value_distances
    ):
        super().__init__()
        self.mds_iters = mds_iters
        self.use_eigen_mds = use_eigen_mds
        self.predict_real_value_distances = predict_real_value_distances

    def forward(
        self,
        *,
        seq,
        distance_pred,
        trunk_embeds,
        N_mask,
        CA_mask,
        C_mask,
        cloud_mask,
        bb_flat_mask_crossed,
        num_atoms
    ):
        if self.predict_real_value_distances:
            distances, distance_std = distance_pred.unbind(dim = -1)
            weights = (1 / (1 + distance_std)) # could also do a distance_std.sigmoid() here
        else:
            distances, weights = center_distogram_torch(distance_pred)

        # set unwanted atoms to weight=0 (like C-beta in glycine)

        if not self.use_eigen_mds:
            weights.masked_fill_( torch.logical_not(bb_flat_mask_crossed), 0.)
        else:
            weights = None

        coords_3d, _ = MDScaling(
            distances,
            weights = weights,
            iters = self.mds_iters,
            fix_mirror = True,
            N_mask = N_mask,
            CA_mask = CA_mask,
            C_mask = C_mask
        )

        coords = rearrange(coords_3d, 'b c n -> b n c')
        # will init all sidechain coords to cbeta if present else c_alpha
        coords = sidechain_container(seq, coords, n_aa = num_atoms, cloud_mask = cloud_mask)
        coords = rearrange(coords, 'b n l d -> b (n l) d')

        return coords

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
        seq_pos_emb = None,
        msa_pos_emb = None,
        seq_to_msa_pos_emb = None,
        msa_to_seq_pos_emb = None,
        **kwargs
    ):
        for ((attn, ff, msa_attn), (cross_attn, msa_ff, msa_cross_attn)) in zip(*[iter(self.blocks)] * 2):

            # self attention

            x = attn(x, shape = seq_shape, mask = mask, rotary_emb = seq_pos_emb) + x

            if exists(m):
                m = msa_attn(m, shape = msa_shape, mask = msa_mask, rotary_emb = msa_pos_emb) + m

                # cross attention

                x = cross_attn(x, m, mask = mask, context_mask = msa_mask, shape = seq_shape, context_shape = msa_shape, rotary_emb = seq_to_msa_pos_emb) + x
                m = msa_cross_attn(m, x, mask = msa_mask, context_mask = mask, shape = msa_shape, context_shape = seq_shape, rotary_emb = msa_to_seq_pos_emb) + m

            # feedforwards

            x = ff(x, shape = seq_shape) + x

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
        attn_types = ('full',),
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
        atoms = 'backbone-only',               # number of atoms to reconstitute each residue to, defaults to 3 for C, C-alpha, N
        predict_angles = False,
        symmetrize_omega = False,
        predict_coords = False,                # structure module related keyword arguments below
        predict_real_value_distances = False,
        trunk_embeds_to_se3_edges = 0,         # feeds pairwise projected logits from the trunk embeddings into the equivariant transformer as edges
        se3_edges_fourier_encodings = 4,       # number of fourier encodings for se3 edges
        return_aux_logits = False,
        template_embedder_type = 'en',         # use E(n) Transformer for embedding templates
        structure_module_type = 'se3',         # uses SE3 Transformer - but if set to false, will use the new E(n)-Transformer
        structure_module_dim = 4,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 4,
        structure_module_refinement_iters = 2,
        structure_module_knn = 2,
        structure_module_adj_neighbors = 2,
        structure_module_adj_dim = 4,
        cross_attn_linear = False,
        cross_attn_linear_projection_update_every = 1000,
        cross_attn_kron_primary = False,
        cross_attn_kron_msa = False,
        disable_token_embed = False,
        disable_cross_attn_rotary = False,
        structure_num_global_nodes = 0,
        mds_iters = 5,                          # mds coupling related parameters
        use_eigen_mds = False,
        coords_module = None                    # custom coords generation module
    ):
        super().__init__()
        self.dim = dim

        # token embedding

        self.token_emb = nn.Embedding(num_tokens, dim) if not disable_token_embed else Always(0)
        self.disable_token_embed = disable_token_embed

        # positional embedding

        self.self_attn_rotary_emb = FixedPositionalEmbedding(dim_head)

        self.disable_cross_attn_rotary = disable_cross_attn_rotary
        self.cross_attn_seq_rotary_emb = AxialRotaryEmbedding(dim_head)
        self.cross_attn_msa_rotary_emb = FixedPositionalEmbedding((dim_head // 4) * 2)

        # template embedding

        self.template_dist_emb = nn.Embedding(constants.DISTOGRAM_BUCKETS, dim)
        self.template_num_pos_emb = nn.Embedding(max_num_templates, dim)

        # projection for angles, if needed

        self.predict_angles = predict_angles
        self.symmetrize_omega = symmetrize_omega

        if predict_angles:
            self.to_prob_theta = nn.Linear(dim, constants.THETA_BUCKETS)
            self.to_prob_phi   = nn.Linear(dim, constants.PHI_BUCKETS)
            self.to_prob_omega = nn.Linear(dim, constants.OMEGA_BUCKETS)

        # when predicting the coordinates, whether to return the other logits, distogram (and optionally, angles)

        self.return_aux_logits = return_aux_logits

        # template sidechain encoding

        self.template_embedder_type = template_embedder_type

        if template_embedder_type == 'se3':
            self.template_sidechain_emb = SE3TemplateEmbedder(
                dim = dim,
                dim_head = dim,
                heads = 1,
                num_neighbors = 12,
                depth = 4,
                input_degrees = 2,
                num_degrees = 2,
                output_degrees = 1,
                reversible = True,
                tie_key_values = True,
                one_headed_key_values = True,
                num_positions = max_seq_len
            )
        elif template_embedder_type == 'en':
            self.template_sidechain_emb = EnTransformer(
                dim = dim,
                dim_head = dim,
                heads = 1,
                neighbors = 32,
                depth = 4
            )
        else:
            raise ValueError('template embedder type must be either "se3" or "en"')

        # custom embedding projection

        self.embedd_project = nn.Linear(num_embedds, dim)

        # attention types

        layers_sparse_attn = cast_tuple(sparse_self_attn, depth)
        layers_cross_attn_linear = cast_tuple(cross_attn_linear, depth)

        # main trunk modules

        prenorm = partial(PreNorm, dim)
        prenorm_cross = partial(PreNormCross, dim)

        layers = nn.ModuleList([])
        attn_types = islice(cycle(attn_types), depth)

        for ind, layer_sparse_attn, layer_cross_attn_linear, attn_type in zip(range(depth), layers_sparse_attn, layers_cross_attn_linear, attn_types):

            # alternate between row and column attention to save memory each layer

            row_attn = ind % 2 == 0
            col_attn = ind % 2 == 1

            # self attention, for main sequence, msa, and optionally, templates

            ff_tensor_slice = (slice(None), slice(0, 1))

            if attn_type == 'full':
                tensor_slice = None
                template_axial_attn = True
            elif attn_type == 'intra_attn':
                tensor_slice = None
                template_axial_attn = False
            elif attn_type == 'seq_only':
                tensor_slice = (slice(None), slice(0, 1))
                template_axial_attn = False
            else:
                raise ValueError(f'cannot find attention type {attn_type}')

            layers.append(nn.ModuleList([
                prenorm(InterceptAxialAttention(tensor_slice, AxialAttention(dim = dim, template_axial_attn = template_axial_attn, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, sparse_attn = sparse_self_attn, row_attn = row_attn, col_attn = col_attn))),
                prenorm(InterceptFeedForward(ff_tensor_slice, ff = LocalFeedForward(dim = dim, hidden_dim = dim * 4, dropout = ff_dropout))),
                prenorm(AxialAttention(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, tie_row_attn = msa_tie_row_attn, row_attn = row_attn, col_attn = col_attn)),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout)),
            ]))

            # cross attention, for main sequence -> msa and then msa -> sequence

            intercept_fn = partial(InterceptAttention, (slice(None), slice(0, 1)))

            if layer_cross_attn_linear:
                cross_attn_fn = lambda: LinearAttention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            else:
                cross_attn_fn = lambda: Attention(dim = dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, compress_ratio = cross_attn_compress_ratio)

            layers.append(nn.ModuleList([
                intercept_fn(context = False, attn = prenorm_cross(KronInputWrapper(cross_attn_fn(), kron_queries = cross_attn_kron_primary, kron_context = cross_attn_kron_msa))),
                prenorm(FeedForward(dim = dim, dropout = ff_dropout)),
                intercept_fn(context = True, attn = prenorm_cross(KronInputWrapper(cross_attn_fn(), kron_context = cross_attn_kron_primary, kron_queries = cross_attn_kron_msa))),
                prenorm(InterceptFeedForward(ff_tensor_slice, LocalFeedForward(dim = dim, hidden_dim = dim * 4, dropout = ff_dropout))),
            ]))

        if not reversible:
            layers = nn.ModuleList(list(map(lambda t: t[:3], layers))) # remove last feed forward if not reversible

        trunk_class = SequentialSequence if not reversible else ReversibleSequence
        self.net = trunk_class(layers)

        # updating linear attention projections, if there exists linear attention

        self.has_linear_attn = any(layers_cross_attn_linear)
        if self.has_linear_attn:
            self.proj_updater = ProjectionUpdater(self.net, cross_attn_linear_projection_update_every)

        # atom masking

        if atoms == 'backbone-only':
            atom_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif atoms == 'backbone-with-cbeta':
            atom_mask = torch.tensor([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif atoms == 'all':
            atom_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        elif atoms == 'backbone-with-oxygen':
            atom_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif atoms == 'backbone-with-cbeta-and-oxygen':
            atom_mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif torch.is_tensor(atoms):
            atom_mask = atoms
        else:
            raise ValueError('atoms needs to be a valid string or a mask tensor of shape (14,) ')

        assert tuple(atom_mask.shape) == (14,), 'atoms needs to be of the correct shape (14,)'

        self.register_buffer('atom_mask', atom_mask.bool())

        # to distogram output

        trunk_upsample_factor = atom_mask.sum().item()
        needs_upsample = trunk_upsample_factor > 1

        self.predict_real_value_distances = predict_real_value_distances
        dim_distance_pred = constants.DISTOGRAM_BUCKETS if not predict_real_value_distances else 2   # 2 for predicting mean and standard deviation values of real-value distance

        self.to_distogram_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Sequential(
                nn.Linear(dim, dim * (trunk_upsample_factor ** 2)),
                Rearrange('b h w c -> b c h w'),
                nn.PixelShuffle(trunk_upsample_factor),
                Rearrange('b c h w -> b h w c')
            ) if needs_upsample else nn.Identity(),
            nn.Linear(dim, dim_distance_pred)
        )

        # coords modules

        self.trunk_to_coords = coords_module

        if not exists(self.trunk_to_coords):
            self.trunk_to_coords = CoordModuleMDS(
                mds_iters,
                use_eigen_mds,
                predict_real_value_distances,
            )

        # global node tokens for se3 structure module

        global_feats_dim = None

        self.global_pool_attns = nn.ModuleList([])
        self.structure_num_global_nodes = structure_num_global_nodes

        if structure_num_global_nodes > 0:
            assert structure_module_type == 'se3', 'se3 transformer must be used in order to use global node token feature'
            self.global_queries = nn.Parameter(torch.randn(structure_num_global_nodes, dim))
            self.global_pool_attns.append(Attention(dim = dim))
            global_feats_dim = dim

        # to coordinate output

        self.predict_coords = predict_coords
        self.mds_iters = mds_iters
        self.structure_module_refinement_iters = structure_module_refinement_iters

        self.trunk_to_structure_dim = nn.Linear(dim, structure_module_dim)

        self.trunk_embeds_to_se3_edges = trunk_embeds_to_se3_edges
        self.se3_edges_fourier_encodings = se3_edges_fourier_encodings
        edge_dim = ((2 * se3_edges_fourier_encodings) + 1) * trunk_embeds_to_se3_edges

        self.to_equivariant_net_edges = nn.Linear(dim, trunk_embeds_to_se3_edges) if trunk_embeds_to_se3_edges > 0 else None

        if self.predict_coords:
            with torch_default_dtype(torch.float64):
                self.structure_module_embeds = nn.Embedding(num_tokens, structure_module_dim)
                self.atom_tokens_embed = nn.Embedding(len(ATOM_IDS), structure_module_dim)

                if structure_module_type == 'se3':
                    self.structure_module = SE3TransformerWrapper(
                        dim = structure_module_dim,
                        depth = structure_module_depth,
                        input_degrees = 1,
                        num_degrees = 3,
                        output_degrees = 2,
                        heads = structure_module_heads,
                        differentiable_coors = True,
                        num_neighbors = 0, # use only bonded neighbors for now
                        attend_sparse_neighbors = True,
                        edge_dim = edge_dim,
                        num_adj_degrees = structure_module_adj_neighbors,
                        adj_dim = structure_module_adj_dim,
                        global_feats_dim = global_feats_dim,
                        tie_key_values = True,
                        one_headed_key_values = True,
                        num_positions = max_seq_len * 14 # hard code as 14 since residual to atom is not flexible atm
                    )
                elif structure_module_type == 'en':
                    self.structure_module = EnTransformer(
                        dim = structure_module_dim,
                        depth = structure_module_depth,
                        heads = structure_module_heads,
                        rel_pos_emb = True,
                        neighbors = 0,
                        only_sparse_neighbors = True,
                        edge_dim = edge_dim,
                        num_adj_degrees = structure_module_adj_neighbors,
                        adj_dim = 4
                    )
                elif structure_module_type == 'egnn':
                    self.structure_module = EGNN_Network(
                        dim = structure_module_dim,
                        depth = structure_module_depth,
                        num_positions = max_seq_len * 14, # hard code as 14 since residual to atom is not flexible atm
                        edge_dim = edge_dim,
                        num_adj_degrees = structure_module_adj_neighbors,
                        adj_dim = structure_module_adj_dim,
                        coor_weights_clamp_value = 2.
                    )
                else:
                    raise ValueError('structure module must be either "se3", "en", or "egnn" for SE3 Transformers, E(n)-Transformers, or EGNN respectively')

        # aux confidence measure
        self.lddt_linear = nn.Linear(structure_module_dim, 1)

    def fix_projections_(self):
        if not self.has_linear_attn:
            return
        self.proj_updater.fix_projections_()

    def forward(
        self,
        seq,
        msa = None,
        mask = None,
        msa_mask = None,
        seq_embed = None,
        msa_embed = None,
        templates_seq = None,
        templates_dist = None,
        templates_mask = None,
        templates_coors = None,
        templates_sidechains = None,
        embedds = None,
        return_trunk = False,
        return_confidence = False
    ):
        assert not (self.disable_token_embed and not exists(seq_embed)), 'sequence embedding must be supplied if one has disabled token embedding'
        assert not (self.disable_token_embed and not exists(msa_embed)), 'msa embedding must be supplied if one has disabled token embedding'

        # update linear projections

        if self.has_linear_attn:
            self.proj_updater.redraw_projections()

        # variables

        b, n, device = *seq.shape[:2], seq.device
        n_range = torch.arange(n, device = device)

        # unpack (AA_code, atom_pos)

        if isinstance(seq, (list, tuple)):
            seq, seq_pos = seq

        # embed main sequence

        x = self.token_emb(seq)

        if exists(seq_embed):
            x += seq_embed

        # outer sum

        x = rearrange(x, 'b i d -> b () i () d') + rearrange(x, 'b j d-> b () () j d') # create pair-wise residue embeds
        x_mask = rearrange(mask, 'b i -> b () i ()') + rearrange(mask, 'b j -> b () () j') if exists(mask) else None

        # embed multiple sequence alignment (msa)

        m = None
        msa_shape = None
        if exists(msa):
            m = self.token_emb(msa)

            if exists(msa_embed):
                m += msa_embed

            msa_shape = m.shape
            m = rearrange(m, 'b m n d -> b (m n) d')

            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, torch.ones_like(msa).bool())

        elif exists(embedds):
            m = self.embedd_project(embedds)

            msa_shape = m.shape
            m = rearrange(m, 'b m n d -> b (m n) d')
            
            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, torch.ones_like(embedds[..., -1]).bool())

        if exists(msa_mask):
            msa_mask = rearrange(msa_mask, 'b m n -> b (m n)')

        # embed templates, if present

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
                if self.template_embedder_type == 'se3':
                    t_seq = self.template_sidechain_emb(
                        t_seq,
                        templates_sidechains,
                        templates_coors,
                        mask = templates_mask
                    )
                elif self.template_embedder_type == 'en':
                    shape = t_seq.shape
                    t_seq = rearrange(t_seq, 'b t n d -> (b t) n d')
                    templates_coors = rearrange(templates_coors, 'b t n c -> (b t) n c')
                    en_mask = rearrange(templates_mask, 'b t n -> (b t) n')

                    t_seq, _ = self.template_sidechain_emb(
                        t_seq,
                        templates_coors,
                        mask = en_mask
                    )

                    t_seq = t_seq.reshape(*shape)

            # embed template distances

            t_dist = self.template_dist_emb(templates_dist)

            t_seq = rearrange(t_seq, 'b t i d -> b t i () d') + rearrange(t_seq, 'b t j d -> b t () j d')
            t = t_seq + t_dist

            # template pos emb

            template_num_pos_emb = self.template_num_pos_emb(torch.arange(num_templates, device = device))
            t += rearrange(template_num_pos_emb, 't d-> () t () () d')

            assert t.shape[-2:] == x.shape[-2:]

            x = torch.cat((x, t), dim = 1)

            if exists(templates_mask):
                t_mask = rearrange(templates_mask, 'b t i -> b t i ()') * rearrange(templates_mask, 'b t j -> b t () j')
                x_mask = torch.cat((x_mask, t_mask), dim = 1)

        # flatten

        seq_shape = x.shape
        x = rearrange(x, 'b t i j d -> b (t i j) d')
        x_mask = rearrange(x_mask, 'b t i j -> b (t i j)') if exists(mask) else None

        # pos emb

        seq_pos_emb = self.self_attn_rotary_emb(n, device = device)

        msa_pos_emb = None
        seq_to_msa_pos_emb = None
        msa_to_seq_pos_emb = None

        if exists(msa):
            num_msa = msa_shape[-3]
            msa_seq_len = msa_shape[-2]

            msa_pos_emb = self.self_attn_rotary_emb(msa_seq_len, device = device)


            if not self.disable_cross_attn_rotary:
                cross_seq_pos_emb = self.cross_attn_seq_rotary_emb(n, device = device)
                cross_msa_pos_emb = self.cross_attn_msa_rotary_emb(msa_seq_len, device = device)

                cross_msa_pos_emb = list(map(lambda t: repeat(t, 'b n d -> b (m n) (r d)', m = num_msa, r = 2), cross_msa_pos_emb))

                seq_to_msa_pos_emb = (cross_seq_pos_emb, cross_msa_pos_emb)
                msa_to_seq_pos_emb = (cross_msa_pos_emb, cross_seq_pos_emb)

        # trunk

        x, m = self.net(
            x,
            m,
            seq_shape,
            msa_shape,
            mask = x_mask,
            msa_mask = msa_mask,
            seq_pos_emb = seq_pos_emb,
            msa_pos_emb = (msa_pos_emb, None),
            seq_to_msa_pos_emb = seq_to_msa_pos_emb,
            msa_to_seq_pos_emb = msa_to_seq_pos_emb
        )

        # remove templates, if present

        x = x.view(seq_shape)
        x = x[:, 0]

        # calculate theta and phi before symmetrization

        if self.predict_angles:
            theta_logits = self.to_prob_theta(x)
            phi_logits = self.to_prob_phi(x)

        # embeds to distogram

        trunk_embeds = (x + rearrange(x, 'b i j d -> b j i d')) * 0.5  # symmetrize
        distance_pred = self.to_distogram_logits(trunk_embeds)

        # determine angles, if specified

        ret = distance_pred

        if self.predict_angles:
            omega_input = trunk_embeds if self.symmetrize_omega else x
            omega_logits = self.to_prob_omega(omega_input)
            ret = Logits(distance_pred, theta_logits, phi_logits, omega_logits)

        if not self.predict_coords or return_trunk:
            return ret

        # prepare atom mask

        atom_mask = self.atom_mask
        num_atoms = atom_mask.sum().item()

        # prepare mask for backbone coordinates

        N_mask, CA_mask, C_mask = scn_backbone_mask(seq, boolean = True, n_aa = num_atoms)

        cloud_mask = scn_cloud_mask(seq, boolean = True)
        flat_cloud_mask = rearrange(cloud_mask, 'b l c -> b (l c)')
        chain_mask = (mask.unsqueeze(-1) * cloud_mask)
        flat_chain_mask = rearrange(chain_mask, 'b l c -> b (l c)')

        bb_flat_mask = rearrange(chain_mask[..., :num_atoms], 'b l c -> b (l c)')
        bb_flat_mask_crossed = rearrange(bb_flat_mask, 'b i -> b i ()') * rearrange(bb_flat_mask, 'b j -> b () j')

        coords = self.trunk_to_coords(
            seq = seq, 
            distance_pred = distance_pred,
            trunk_embeds = trunk_embeds,
            N_mask = N_mask,
            CA_mask = CA_mask,
            C_mask = C_mask,
            cloud_mask = cloud_mask,
            bb_flat_mask_crossed = bb_flat_mask_crossed,
            num_atoms = num_atoms
        )

        # derive nodes

        num_atoms_per_residue = 14

        structure_embed = self.trunk_to_structure_dim(trunk_embeds)
        x = reduce(structure_embed, 'b i j d -> b i d', 'mean')
        x += self.structure_module_embeds(seq)
        x = repeat(x, 'b n d -> b n l d', l = num_atoms_per_residue)

        atom_tokens = scn_atom_embedd(seq)
        x += self.atom_tokens_embed(atom_tokens)

        x = rearrange(x, 'b n l d -> b (n l) d')

        # derive edges from trunk -> equivariant network, if needed

        edges = None
        if exists(self.to_equivariant_net_edges):
            edges = self.to_equivariant_net_edges(trunk_embeds)
            edges = fourier_encode(edges, num_encodings = self.se3_edges_fourier_encodings, include_self = True)
            edges = repeat(edges, 'b i j d -> b (i l1) (j l2) d', l1 = num_atoms_per_residue, l2 = num_atoms_per_residue)

        # derive adjacency matrix
        # todo - fix so Cbeta is connected correctly

        adj_idxs, adj_num = prot_covalent_bond(seq, adj_degree=1, cloud_mask=cloud_mask)
        adj_mat = adj_num.bool()

        # derive global features

        structure_kwargs = {}
        pooled_feats = None
        if self.structure_num_global_nodes > 0:
            pooled_feats = repeat(self.global_queries, 'n d -> b n d', b = b)
            to_pool = rearrange(trunk_embeds, 'b ... d -> b (...) d')

            for attn in self.global_pool_attns:
                pooled_feats = attn(pooled_feats, context = to_pool, context_mask = x_mask) + pooled_feats

            pooled_feats = pooled_feats.double()
            structure_kwargs = {'global_feats': pooled_feats}

        # prepare only atoms defined by atom mask

        atom_mask = repeat(atom_mask, 'a -> () (n a)', n = n)
        atom_mask_crossed = atom_mask[:, :, None] & atom_mask[:, None, :]
        total_atoms = num_atoms * n

        coords  = coords.masked_select(atom_mask[..., None]).reshape(b, total_atoms, -1)
        x       = x.masked_select(atom_mask[..., None]).reshape(b, total_atoms, -1)
        adj_mat = adj_mat.masked_select(atom_mask_crossed).reshape(b, total_atoms, total_atoms)
        flat_chain_mask = flat_chain_mask.masked_select(atom_mask).reshape(b, total_atoms)

        if exists(edges):
            edges   = edges.masked_select(atom_mask_crossed[..., None]).reshape(b, total_atoms, total_atoms, -1)

        # prepare float64 precision for equivariance

        original_dtype = coords.dtype
        x, coords, edges = map(maybe(lambda t: t.double()), (x, coords, edges))

        # iterative refinement with equivariant transformer in high precision

        with torch_default_dtype(torch.float64):
            for _ in range(self.structure_module_refinement_iters):
                x, coords = self.structure_module(
                    x,
                    coords,
                    mask = flat_chain_mask,
                    adj_mat = adj_mat,
                    edges = edges,
                    **structure_kwargs
                )

        coords.type(original_dtype)

        if self.return_aux_logits:
            return coords, ret

        if return_confidence:
            return coords, self.lddt_linear(x.float())

        return coords
