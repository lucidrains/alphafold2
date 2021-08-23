import torch
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from inspect import isfunction
from functools import partial
from dataclasses import dataclass
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from alphafold2_pytorch.utils import *
import alphafold2_pytorch.constants as constants
from alphafold2_pytorch.mlm import MLM

# structure module

from invariant_point_attention import IPABlock
from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix

# constants

@dataclass
class Recyclables:
    coords: torch.Tensor
    single_msa_repr_row: torch.Tensor
    pairwise_repr: torch.Tensor

@dataclass
class ReturnValues:
    distance: torch.Tensor = None
    theta: torch.Tensor = None
    phi: torch.Tensor = None
    omega: torch.Tensor = None
    msa_mlm_loss: torch.Tensor = None
    recyclables: Recyclables = None

# helpers

def exists(val):
    return val is not None

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

    def forward(self, x, mask = None, attn_bias = None, context = None, context_mask = None, tie_dim = None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # query / key similarities

        if exists(tie_dim):
            # as in the paper, for the extra MSAs
            # they average the queries along the rows of the MSAs
            # they named this particular module MSAColumnGlobalAttention

            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r = tie_dim), (q, k))
            q = q.mean(dim = 1)

            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)

        if exists(attn_bias):
            dots = dots + attn_bias

        # masking

        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2], device = device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention

        dots = dots - dots.max(dim = -1, keepdims = True).values
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating

        gates = self.gating(x)
        out = out * gates.sigmoid()

        # combine to out

        out = self.to_out(out)
        return out

class AxialAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        row_attn = True,
        col_attn = True,
        accept_edges = False,
        global_query_attn = False,
        **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn

        self.norm = nn.LayerNorm(dim)

        self.attn = Attention(dim = dim, heads = heads, **kwargs)

        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None

    def forward(self, x, edges = None, mask = None):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'

        b, h, w, d = x.shape

        x = self.norm(x)

        # axial attention

        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'

        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'

        x = rearrange(x, input_fold_eq)

        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = axial_dim)

        tie_dim = axial_dim if self.global_query_attn else None

        out = self.attn(x, mask = mask, attn_bias = attn_bias, tie_dim = tie_dim)
        out = rearrange(out, output_fold_eq, h = h, w = w)

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

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
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
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)

# evoformer blocks

class OuterMean(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim = None,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        hidden_dim = default(hidden_dim, dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d')

        if exists(mask):
            # masked mean, if there are padding in the rows of the MSA
            mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(mask, 'b m j -> b m () j ()')
            outer = outer.masked_fill(~mask, 0.)
            outer = outer.mean(dim = 1) / (mask.sum(dim = 1) + self.eps)
        else:
            outer = outer.mean(dim = 1)

        return self.proj_out(outer)

class PairwiseAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.,
        global_column_attn = False
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim)

        self.triangle_attention_outgoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.triangle_attention_ingoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True, global_query_attn = global_column_attn)
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim = dim, mix = 'outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim = dim, mix = 'ingoing')

    def forward(
        self,
        x,
        mask = None,
        msa_repr = None,
        msa_mask = None
    ):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr, mask = msa_mask)

        x = self.triangle_multiply_outgoing(x, mask = mask) + x
        x = self.triangle_multiply_ingoing(x, mask = mask) + x
        x = self.triangle_attention_outgoing(x, edges = x, mask = mask) + x
        x = self.triangle_attention_ingoing(x, edges = x, mask = mask) + x
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
        self.row_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.col_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True)

    def forward(
        self,
        x,
        mask = None,
        pairwise_repr = None
    ):
        x = self.row_attn(x, mask = mask, edges = pairwise_repr) + x
        x = self.col_attn(x, mask = mask) + x
        return x

# main evoformer class

class EvoformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        global_column_attn = False
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            PairwiseAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, global_column_attn = global_column_attn),
            FeedForward(dim = dim, dropout = ff_dropout),
            MsaAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
            FeedForward(dim = dim, dropout = ff_dropout),
        ])

    def forward(self, inputs):
        x, m, mask, msa_mask = inputs
        attn, ff, msa_attn, msa_ff = self.layer

        # msa attention and transition

        m = msa_attn(m, mask = msa_mask, pairwise_repr = x)
        m = msa_ff(m) + m

        # pairwise attention and transition

        x = attn(x, mask = mask, msa_repr = m, msa_mask = msa_mask)
        x = ff(x) + x

        return x, m, mask, msa_mask

class Evoformer(nn.Module):
    def __init__(
        self,
        *,
        depth,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([EvoformerBlock(**kwargs) for _ in range(depth)])

    def forward(
        self,
        x,
        m,
        mask = None,
        msa_mask = None
    ):
        inp = (x, m, mask, msa_mask)
        x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
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
        extra_msa_evoformer_layers = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        templates_dim = 32,
        templates_embed_layers = 4,
        templates_angles_feats_dim = 55,
        predict_angles = False,
        symmetrize_omega = False,
        predict_coords = False,                # structure module related keyword arguments below
        structure_module_depth = 4,
        structure_module_heads = 1,
        structure_module_dim_head = 4,
        disable_token_embed = False,
        mlm_mask_prob = 0.15,
        mlm_random_replace_token_prob = 0.1,
        mlm_keep_token_same_prob = 0.1,
        mlm_exclude_token_ids = (0,),
        recycling_distance_buckets = 32
    ):
        super().__init__()
        self.dim = dim

        # token embedding

        self.token_emb = nn.Embedding(num_tokens + 1, dim) if not disable_token_embed else Always(0)
        self.to_pairwise_repr = nn.Linear(dim, dim * 2)
        self.disable_token_embed = disable_token_embed

        # positional embedding

        self.max_rel_dist = max_rel_dist
        self.pos_emb = nn.Embedding(max_rel_dist * 2 + 1, dim)

        # extra msa embedding

        self.extra_msa_evoformer = Evoformer(
            dim = dim,
            depth = extra_msa_evoformer_layers,
            seq_len = max_seq_len,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            global_column_attn = True
        )

        # template embedding

        self.to_template_embed = nn.Linear(templates_dim, dim)
        self.templates_embed_layers = templates_embed_layers

        self.template_pairwise_embedder = PairwiseAttentionBlock(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            seq_len = max_seq_len
        )

        self.template_pointwise_attn = Attention(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            dropout = attn_dropout
        )

        self.template_angle_mlp = nn.Sequential(
            nn.Linear(templates_angles_feats_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # projection for angles, if needed

        self.predict_angles = predict_angles
        self.symmetrize_omega = symmetrize_omega

        if predict_angles:
            self.to_prob_theta = nn.Linear(dim, constants.THETA_BUCKETS)
            self.to_prob_phi   = nn.Linear(dim, constants.PHI_BUCKETS)
            self.to_prob_omega = nn.Linear(dim, constants.OMEGA_BUCKETS)

        # custom embedding projection

        self.embedd_project = nn.Linear(num_embedds, dim)

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

        # MSA SSL MLM

        self.mlm = MLM(
            dim = dim,
            num_tokens = num_tokens,
            mask_id = num_tokens, # last token of embedding is used for masking
            mask_prob = mlm_mask_prob,
            keep_token_same_prob = mlm_keep_token_same_prob,
            random_replace_token_prob = mlm_random_replace_token_prob,
            exclude_token_ids = mlm_exclude_token_ids
        )

        # calculate distogram logits

        self.to_distogram_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, constants.DISTOGRAM_BUCKETS)
        )

        # to coordinate output

        self.predict_coords = predict_coords
        self.structure_module_depth = structure_module_depth

        self.msa_to_single_repr_dim = nn.Linear(dim, dim)
        self.trunk_to_pairwise_repr_dim = nn.Linear(dim, dim)

        with torch_default_dtype(torch.float32):
            self.ipa_block = IPABlock(
                dim = dim,
                heads = structure_module_heads,
            )

            self.to_quaternion_update = nn.Linear(dim, 6)

        self.to_points = nn.Linear(dim, 3)

        # aux confidence measure

        self.lddt_linear = nn.Linear(dim, 1)

        # recycling params

        self.recycling_msa_norm = nn.LayerNorm(dim)
        self.recycling_pairwise_norm = nn.LayerNorm(dim)
        self.recycling_distance_embed = nn.Embedding(recycling_distance_buckets, dim)
        self.recycling_distance_buckets = recycling_distance_buckets

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
        recyclables = None,
        return_trunk = False,
        return_confidence = False,
        return_recyclables = False,
        return_aux_logits = False
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

        # mlm for MSAs

        if self.training and exists(msa):
            original_msa = msa
            msa_mask = default(msa_mask, lambda: torch.ones_like(msa).bool())

            noised_msa, replaced_msa_mask = self.mlm.noise(msa, msa_mask)
            msa = noised_msa

        # embed multiple sequence alignment (msa)

        if exists(msa):
            m = self.token_emb(msa)

            if exists(msa_embed):
                m = m + msa_embed

            # add single representation to msa representation

            m = m + rearrange(x, 'b n d -> b () n d')

            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, lambda: torch.ones_like(msa).bool())

        elif exists(embedds):
            m = self.embedd_project(embedds)
            
            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, lambda: torch.ones_like(embedds[..., -1]).bool())
        else:
            raise Error('either MSA or embeds must be given')

        # derive pairwise representation

        x_left, x_right = self.to_pairwise_repr(x).chunk(2, dim = -1)
        x = rearrange(x_left, 'b i d -> b i () d') + rearrange(x_right, 'b j d-> b () j d') # create pair-wise residue embeds
        x_mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j') if exists(mask) else None

        # add relative positional embedding

        seq_index = default(seq_index, lambda: torch.arange(n, device = device))
        seq_rel_dist = rearrange(seq_index, 'i -> () i ()') - rearrange(seq_index, 'j -> () () j')
        seq_rel_dist = seq_rel_dist.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        rel_pos_emb = self.pos_emb(seq_rel_dist)

        x = x + rel_pos_emb

        # add recyclables, if present

        if exists(recyclables):
            m[:, 0] = m[:, 0] + self.recycling_msa_norm(recyclables.single_msa_repr_row)
            x = x + self.recycling_pairwise_norm(recyclables.pairwise_repr)

            distances = torch.cdist(recyclables.coords, recyclables.coords, p=2)
            boundaries = torch.linspace(2, 20, steps = self.recycling_distance_buckets, device = device)
            discretized_distances = torch.bucketize(distances, boundaries[:-1])
            distance_embed = self.recycling_distance_embed(discretized_distances)

            x = x + distance_embed

        # embed templates, if present

        if exists(templates_feats):
            _, num_templates, *_ = templates_feats.shape

            # embed template

            t = self.to_template_embed(templates_feats)
            t_mask_crossed = rearrange(templates_mask, 'b t i -> b t i ()') * rearrange(templates_mask, 'b t j -> b t () j')

            t = rearrange(t, 'b t ... -> (b t) ...')
            t_mask_crossed = rearrange(t_mask_crossed, 'b t ... -> (b t) ...')

            for _ in range(self.templates_embed_layers):
                t = self.template_pairwise_embedder(t, mask = t_mask_crossed)

            t = rearrange(t, '(b t) ... -> b t ...', t = num_templates)
            t_mask_crossed = rearrange(t_mask_crossed, '(b t) ... -> b t ...', t = num_templates)

            # template pos emb

            x_point = rearrange(x, 'b i j d -> (b i j) () d')
            t_point = rearrange(t, 'b t i j d -> (b i j) t d')
            x_mask_point = rearrange(x_mask, 'b i j -> (b i j) ()')
            t_mask_point = rearrange(t_mask_crossed, 'b t i j -> (b i j) t')

            template_pooled = self.template_pointwise_attn(
                x_point,
                context = t_point,
                mask = x_mask_point,
                context_mask = t_mask_point
            )

            template_pooled_mask = rearrange(t_mask_point.sum(dim = -1) > 0, 'b -> b () ()')
            template_pooled = template_pooled * template_pooled_mask

            template_pooled = rearrange(template_pooled, '(b i j) () d -> b i j d', i = n, j = n)
            x = x + template_pooled

        # add template angle features to MSAs by passing through MLP and then concat

        if exists(templates_angles):
            t_angle_feats = self.template_angle_mlp(templates_angles)
            m = torch.cat((m, t_angle_feats), dim = 1)
            msa_mask = torch.cat((msa_mask, templates_mask), dim = 1)

        # embed extra msa, if present

        if exists(extra_msa):
            extra_m = self.token_emb(msa)
            extra_msa_mask = default(extra_msa_mask, torch.ones_like(extra_m).bool())

            x, extra_m = self.extra_msa_evoformer(
                x,
                extra_m,
                mask = x_mask,
                msa_mask = extra_msa_mask
            )

        # trunk

        x, m = self.net(
            x,
            m,
            mask = x_mask,
            msa_mask = msa_mask
        )

        # ready output container

        ret = ReturnValues()

        # calculate theta and phi before symmetrization

        if self.predict_angles:
            ret.theta_logits = self.to_prob_theta(x)
            ret.phi_logits = self.to_prob_phi(x)

        # embeds to distogram

        trunk_embeds = (x + rearrange(x, 'b i j d -> b j i d')) * 0.5  # symmetrize
        distance_pred = self.to_distogram_logits(trunk_embeds)
        ret.distance = distance_pred

        # calculate mlm loss, if training

        msa_mlm_loss = None
        if self.training and exists(msa):
            num_msa = original_msa.shape[1]
            msa_mlm_loss = self.mlm(m[:, :num_msa], original_msa, replaced_msa_mask)

        # determine angles, if specified

        if self.predict_angles:
            omega_input = trunk_embeds if self.symmetrize_omega else x
            ret.omega_logits = self.to_prob_omega(omega_input)

        if not self.predict_coords or return_trunk:
            return ret

        # derive single and pairwise embeddings for structural refinement

        single_msa_repr_row = m[:, 0]

        single_repr = self.msa_to_single_repr_dim(single_msa_repr_row)
        pairwise_repr = self.trunk_to_pairwise_repr_dim(x)

        # prepare float32 precision for equivariance

        original_dtype = single_repr.dtype
        single_repr, pairwise_repr = map(lambda t: t.float(), (single_repr, pairwise_repr))

        # iterative refinement with equivariant transformer in high precision

        with torch_default_dtype(torch.float32):

            quaternions = torch.tensor([1., 0., 0., 0.], device = device) # initial rotations
            quaternions = repeat(quaternions, 'd -> b n d', b = b, n = n)
            translations = torch.zeros((b, n, 3), device = device)

            # go through the layers and apply invariant point attention and feedforward

            for _ in range(self.structure_module_depth):

                # the detach comes from
                # https://github.com/deepmind/alphafold/blob/0bab1bf84d9d887aba5cfb6d09af1e8c3ecbc408/alphafold/model/folding.py#L383
                rotations = quaternion_to_matrix(quaternions).detach()

                single_repr = self.ipa_block(
                    single_repr,
                    mask = mask,
                    pairwise_repr = pairwise_repr,
                    rotations = rotations,
                    translations = translations
                )

                # update quaternion and translation

                quaternion_update, translation_update = self.to_quaternion_update(single_repr).chunk(2, dim = -1)
                quaternion_update = F.pad(quaternion_update, (1, 0), value = 1.)

                quaternions = quaternion_multiply(quaternions, quaternion_update)
                translations = translations + einsum('b n c, b n c r -> b n r', translation_update, rotations)

            points_local = self.to_points(single_repr)
            rotations = quaternion_to_matrix(quaternions)
            coords = einsum('b n c, b n c d -> b n d', points_local, rotations) + translations

        coords.type(original_dtype)

        if return_recyclables:
            coords, single_msa_repr_row, pairwise_repr = map(torch.detach, (coords, single_msa_repr_row, pairwise_repr))
            ret.recyclables = Recyclables(coords, single_msa_repr_row, pairwise_repr)

        if return_aux_logits:
            return coords, ret

        if return_confidence:
            return coords, self.lddt_linear(single_repr.float())

        return coords
