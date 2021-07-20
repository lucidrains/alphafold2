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
from alphafold2_pytorch.rotary import DepthWiseConv1d, AxialRotaryEmbedding, FixedPositionalEmbedding, apply_rotary_pos_emb

# structure module

from en_transformer import EnTransformer
from invariant_point_attention import IPATransformer

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

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

# helper classes

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return self.val

# feed forward

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
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)
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
        gating = True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, rotary_emb = None, attn_bias = None, **kwargs):
        device, orig_shape, h = x.device, x.shape, self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # rotary relative positional encoding

        if exists(rotary_emb):
            rot_q, rot_k = cast_tuple(rotary_emb, 2)
            q = apply_rotary_pos_emb(q, rot_q)
            k = apply_rotary_pos_emb(k, rot_k)

        # query / key similarities

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # add attention bias, if supplied (for pairwise to msa attention communication)

        if exists(attn_bias):
            dots = dots + attn_bias

        # masking

        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating

        gates = self.gating(x)
        out = out * gates            

        # combine to out

        out = self.to_out(out)
        return out

class AxialAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        sparse_attn = False,
        row_attn = True,
        col_attn = True,
        accept_edges = False,
        **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'

        attn_class = SparseAttention if sparse_attn else Attention

        self.norm = nn.LayerNorm(dim)

        self.attn_width = attn_class(dim = dim, heads = heads, **kwargs)
        self.attn_height = attn_class(dim = dim, heads = heads, **kwargs)

        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None

        self.row_attn = row_attn
        self.col_attn = col_attn

    def forward(self, x, edges = None, mask = None, rotary_emb = None):
        b, h, w, d = x.shape

        x = self.norm(x)

        # axial mask

        w_mask = h_mask = None

        if exists(mask):
            mask = mask.reshape(b, h, w)
            w_mask = rearrange(mask, 'b h w -> (b w) h')
            h_mask = rearrange(mask, 'b h w -> (b h) w')

        # axial pos emb

        h_rotary_emb, w_rotary_emb = cast_tuple(rotary_emb, 2)

        # axial attention

        out = 0
        axial_attn_count = 0

        if self.row_attn:
            w_x = rearrange(x, 'b h w d -> (b w) h d')
            w_out = self.attn_width(w_x, mask = w_mask, rotary_emb = w_rotary_emb)
            w_out = rearrange(w_out, '(b w) h d -> b h w d', h = h, w = w)

            out += w_out
            axial_attn_count += 1

        if self.col_attn:
            attn_bias = None
            if exists(self.edges_to_attn_bias) and exists(edges):
                attn_bias = self.edges_to_attn_bias(edges)
                attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = h)

            h_x = rearrange(x, 'b h w d -> (b h) w d')
            h_out = self.attn_height(h_x, mask = h_mask, rotary_emb = h_rotary_emb, attn_bias = attn_bias)
            h_out = rearrange(h_out, '(b h) w d -> b h w d', h = h, w = w)

            out += h_out
            axial_attn_count += 1

        out /= axial_attn_count

        return out

class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        mix = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'ingoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'outgoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * left_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)

# evoformer blocks

class OuterMean(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim = None
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden_dim = default(hidden_dim, dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, 'b m i d -> b m i () d') + rearrange(right, 'b m j d -> b m () j d')
        outer = outer.mean(dim = 1)
        return self.proj_out(outer)

class PairwiseAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim)

        self.triangle_attention_ingoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False)
        self.triangle_attention_outgoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True)
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim = dim, mix = 'ingoing')
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim = dim, mix = 'outgoing')

    def forward(
        self,
        x,
        mask = None,
        msa_repr = None,
        rotary_emb = None
    ):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr)

        x = self.triangle_attention_ingoing(x, mask = mask) + x
        x = self.triangle_attention_outgoing(x, mask = mask) + x
        x = self.triangle_multiply_ingoing(x, mask = mask) + x
        x = self.triangle_multiply_outgoing(x, mask = mask) + x
        return x

class MsaAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout
    ):
        super().__init__()
        self.row_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False)
        self.col_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True)

    def forward(
        self,
        x,
        mask = None,
        pairwise_repr = None,
        rotary_emb = None
    ):
        x = self.row_attn(x) + x
        x = self.col_attn(x, edges = pairwise_repr) + x
        return x

# main class

class Evoformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PairwiseAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout),
                MsaAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout),
            ]))

    def forward(
        self,
        x,
        m,
        seq_shape = None,
        msa_shape = None,
        mask = None,
        msa_mask = None,
        seq_pos_emb = None,
        msa_pos_emb = None
    ):
        for attn, ff, msa_attn, msa_ff in self.layers:
            # msa attention and transition

            m = msa_attn(m, mask = msa_mask, pairwise_repr = x, rotary_emb = msa_pos_emb) + m
            m = msa_ff(m) + m

            # pairwise attention and transition

            x = attn(x, mask = mask, rotary_emb = seq_pos_emb) + x
            x = ff(x) + x

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
        sparse_self_attn = False,
        template_attn_depth = 2,
        predict_angles = False,
        symmetrize_omega = False,
        predict_coords = False,                # structure module related keyword arguments below
        return_aux_logits = False,
        structure_module_dim = 4,
        structure_module_depth = 4,
        structure_module_heads = 1,
        structure_module_dim_head = 4,
        disable_token_embed = False
    ):
        super().__init__()
        self.dim = dim

        # token embedding

        self.token_emb = nn.Embedding(num_tokens, dim) if not disable_token_embed else Always(0)
        self.disable_token_embed = disable_token_embed

        # positional embedding

        self.self_attn_rotary_emb = FixedPositionalEmbedding(dim_head)

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

        # template sidechain encoding - only if needed

        self.template_sidechain_emb = EnTransformer(
            dim = dim,
            dim_head = dim,
            heads = 1,
            neighbors = 32,
            depth = 4
        )

        # custom embedding projection

        self.embedd_project = nn.Linear(num_embedds, dim)

        # attention types

        layers_sparse_attn = cycle(cast_tuple(sparse_self_attn))

        # main trunk modules

        self.net = Evoformer(
            dim = dim,
            depth = depth,
            seq_len = max_seq_len,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # calculate distogram logits

        self.to_distogram_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, constants.DISTOGRAM_BUCKETS)
        )

        # to coordinate output

        self.predict_coords = predict_coords

        self.msa_to_single_repr_dim = nn.Linear(dim, structure_module_dim)
        self.trunk_to_pairwise_repr_dim = nn.Linear(dim, structure_module_dim)

        with torch_default_dtype(torch.float32):
            self.ipa_transformer = IPATransformer(
                dim = structure_module_dim,
                depth = structure_module_depth,
                heads = structure_module_heads,
                predict_points = True
            )

        # aux confidence measure

        self.lddt_linear = nn.Linear(structure_module_dim, 1)

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
        return_confidence = False,
        refine = True
    ):
        assert not (self.disable_token_embed and not exists(seq_embed)), 'sequence embedding must be supplied if one has disabled token embedding'
        assert not (self.disable_token_embed and not exists(msa_embed)), 'msa embedding must be supplied if one has disabled token embedding'

        # if MSA is not passed in, just use the sequence itself

        if not exists(msa):
            msa = rearrange(seq, 'b n -> b () n')
            msa_mask = rearrange(mask, 'b n -> b () n')

        # assert on sequence length

        assert msa.shape[-1] == seq.shape[-1], 'sequence length of MSA and primary sequence must be the same'

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

        x = rearrange(x, 'b i d -> b i () d') + rearrange(x, 'b j d-> b () j d') # create pair-wise residue embeds
        x_mask = rearrange(mask, 'b i -> b i ()') + rearrange(mask, 'b j -> b () j') if exists(mask) else None

        # embed multiple sequence alignment (msa)

        m = None
        msa_shape = None
        if exists(msa):
            m = self.token_emb(msa)

            if exists(msa_embed):
                m += msa_embed

            msa_shape = m.shape

            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, torch.ones_like(msa).bool())

        elif exists(embedds):
            m = self.embedd_project(embedds)

            msa_shape = m.shape
            
            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, torch.ones_like(embedds[..., -1]).bool())

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
            x = x + t.mean(dim = 1)

        # flatten

        seq_shape = x.shape

        # pos emb

        seq_pos_emb = self.self_attn_rotary_emb(n, device = device)

        msa_pos_emb = None
        seq_to_msa_pos_emb = None
        msa_to_seq_pos_emb = None

        if exists(msa):
            num_msa = msa_shape[-3]
            msa_seq_len = msa_shape[-2]

            msa_pos_emb = self.self_attn_rotary_emb(msa_seq_len, device = device)

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
        )

        # remove templates, if present

        x = x.view(seq_shape)

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

        # derive single and pairwise embeddings for structural refinement

        m = m.reshape(msa_shape).mean(dim = 1)

        single_repr = self.msa_to_single_repr_dim(m)
        pairwise_repr = self.trunk_to_pairwise_repr_dim(x)

        # prepare float32 precision for equivariance

        original_dtype = single_repr.dtype
        single_repr, pairwise_repr = map(lambda t: t.float(), (single_repr, pairwise_repr))

        # iterative refinement with equivariant transformer in high precision

        with torch_default_dtype(torch.float32):
            coords = self.ipa_transformer(
                single_repr,
                pairwise_repr = pairwise_repr,
                mask = mask
            )

        coords.type(original_dtype)

        if self.return_aux_logits:
            return coords, ret

        if return_confidence:
            return coords, self.lddt_linear(single_repr.float())

        return coords
