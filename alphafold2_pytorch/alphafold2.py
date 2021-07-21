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

from alphafold2_pytorch.utils import *
import alphafold2_pytorch.constants as constants

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

    def forward(self, x, mask = None, attn_bias = None, **kwargs):
        device, orig_shape, h = x.device, x.shape, self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

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
            nn.Linear(dim, heads, bias = False),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None

        self.row_attn = row_attn
        self.col_attn = col_attn

    def forward(self, x, edges = None, mask = None):
        b, h, w, d = x.shape

        x = self.norm(x)

        # axial mask

        w_mask = h_mask = None

        if exists(mask):
            mask = mask.reshape(b, h, w)
            w_mask = rearrange(mask, 'b h w -> (b w) h')
            h_mask = rearrange(mask, 'b h w -> (b h) w')

        # calculate attention bias

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)

        # axial attention

        out = 0
        axial_attn_count = 0

        if self.row_attn:
            w_x = rearrange(x, 'b h w d -> (b w) h d')
            if exists(attn_bias):
                attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = w)

            w_out = self.attn_width(w_x, mask = w_mask, attn_bias = attn_bias)
            w_out = rearrange(w_out, '(b w) h d -> b h w d', h = h, w = w)

            out += w_out
            axial_attn_count += 1

        if self.col_attn:
            h_x = rearrange(x, 'b h w d -> (b h) w d')
            if exists(attn_bias):
                attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = h)

            h_out = self.attn_height(h_x, mask = h_mask, attn_bias = attn_bias)
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
        dropout = 0.
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim)

        self.triangle_attention_ingoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.triangle_attention_outgoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True)
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim = dim, mix = 'ingoing')
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim = dim, mix = 'outgoing')

    def forward(
        self,
        x,
        mask = None,
        msa_repr = None
    ):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr)

        x = self.triangle_attention_ingoing(x, edges = x, mask = mask) + x
        x = self.triangle_attention_outgoing(x, edges = x, mask = mask) + x
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
        dropout = 0.
    ):
        super().__init__()
        self.row_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False)
        self.col_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True)

    def forward(
        self,
        x,
        mask = None,
        pairwise_repr = None
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
        mask = None,
        msa_mask = None
    ):
        for attn, ff, msa_attn, msa_ff in self.layers:
            # msa attention and transition

            m = msa_attn(m, mask = msa_mask, pairwise_repr = x) + m
            m = msa_ff(m) + m

            # pairwise attention and transition

            x = attn(x, mask = mask) + x
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
        max_rel_dist = 32,
        num_tokens = constants.NUM_AMINO_ACIDS,
        num_embedds = constants.NUM_EMBEDDS_TR,
        max_num_msas = constants.MAX_NUM_MSA,
        max_num_templates = constants.MAX_NUM_TEMPLATES,
        attn_dropout = 0.,
        ff_dropout = 0.,
        sparse_self_attn = False,
        template_dim = 32,
        template_embed_layers = 4,
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
        self.to_pairwise_repr = nn.Linear(dim, dim * 2)
        self.disable_token_embed = disable_token_embed

        # positional embedding

        self.max_rel_dist = max_rel_dist
        self.pos_emb = nn.Embedding(max_rel_dist * 2 + 1, dim)

        # template embedding

        self.to_template_embed = nn.Linear(template_dim, dim)
        self.template_embed_layers = template_embed_layers

        self.template_pairwise_embedder = PairwiseAttentionBlock(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            seq_len = max_seq_len
        )

        # projection for angles, if needed

        self.predict_angles = predict_angles
        self.symmetrize_omega = symmetrize_omega

        if predict_angles:
            self.to_prob_theta = nn.Linear(dim, constants.THETA_BUCKETS)
            self.to_prob_phi   = nn.Linear(dim, constants.PHI_BUCKETS)
            self.to_prob_omega = nn.Linear(dim, constants.OMEGA_BUCKETS)

        # when predicting the coordinates, whether to return the other logits, distogram (and optionally, angles)

        self.return_aux_logits = return_aux_logits

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
        extra_msa = None,
        extra_msa_mask = None,
        seq_index = None,
        seq_embed = None,
        msa_embed = None,
        templates_feats = None,
        templates_mask = None,
        templates_angles = None,
        embedds = None,
        return_trunk = False,
        return_confidence = False        
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

        # embed multiple sequence alignment (msa)

        if exists(msa):
            m = self.token_emb(msa)

            if exists(msa_embed):
                m = m + msa_embed

            # add single representation to msa representation

            m = m + rearrange(x, 'b n d -> b () n d')

            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, torch.ones_like(msa).bool())

        elif exists(embedds):
            m = self.embedd_project(embedds)
            
            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, torch.ones_like(embedds[..., -1]).bool())
        else:
            raise Error('either MSA or embeds must be given')

        # derive pairwise representation

        x_left, x_right = self.to_pairwise_repr(x).chunk(2, dim = -1)
        x = rearrange(x_left, 'b i d -> b i () d') + rearrange(x_right, 'b j d-> b () j d') # create pair-wise residue embeds
        x_mask = rearrange(mask, 'b i -> b i ()') + rearrange(mask, 'b j -> b () j') if exists(mask) else None

        # add relative positional embedding

        seq_index = default(seq_index, torch.arange(n, device = device))
        seq_rel_dist = rearrange(seq_index, 'i -> () i ()') - rearrange(seq_index, 'j -> () () j')
        seq_rel_dist = seq_rel_dist.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        rel_pos_emb = self.pos_emb(seq_rel_dist)

        # embed templates, if present

        if exists(templates_feats):
            _, num_templates, *_ = templates_feats.shape

            # embed template

            t = self.to_template_embed(templates_feats)
            templates_mask_crossed = rearrange(templates_mask, 'b t i -> b t i ()') * rearrange(templates_mask, 'b t j -> b t () j')

            t = rearrange(t, 'b t ... -> (b t) ...')
            templates_mask_crossed = rearrange(templates_mask_crossed, 'b t ... -> (b t) ...')

            for _ in range(self.template_embed_layers):
                t = self.template_pairwise_embedder(t, mask = templates_mask_crossed) + t

            t = rearrange(t, '(b t) ... -> b t ...', t = num_templates)
            templates_mask_crossed = rearrange(templates_mask_crossed, '(b t) ... -> b t ...', t = num_templates)

            # template pos emb

            assert t.shape[-2:] == x.shape[-2:]
            x = x + t.mean(dim = 1)

        # trunk

        x, m = self.net(
            x,
            m,
            mask = x_mask,
            msa_mask = msa_mask
        )

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

        m = m.mean(dim = 1)

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
