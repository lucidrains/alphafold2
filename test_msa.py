from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from torch.utils.data import Dataset

from alphafold2_pytorch import Alphafold2, constants
from alphafold2_pytorch.alphafold2 import FeedForward, PreNorm, exists, partial

# n: 20171106 : changed-to: '.' = 0
AA = {
    '.': 32,
    'A': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'V': 18,
    'W': 19,
    'Y': 20,
    'B': 21,
    'J': 21,
    'O': 21,
    'U': 21,
    'X': 21,
    'Z': 21,
}

GRADIENT_ACCUMULATE_EVERY = 16
LEARNING_RATE = 3e-5
NUM_SS_CLASS = 3

from math import ceil

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum, nn

# helper functions


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


# main class


class NystromAttention(nn.Module):
    def __init__(self, in_dim, out_dim, dim_head=64, heads=8, m=256, pinv_iterations=6, residual=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.m = m
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(in_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim)

        self.residual = residual
        if residual:
            self.res_conv = nn.Conv2d(heads, heads, 1, groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        print(f"pre_forward_Nystrom_input: x-> {x.shape}")
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.m, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, 0, padding), value=0)

            if exists(mask):
                mask = F.pad(mask, (0, padding), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q *= self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        attn = attn1 @ attn2_inv @ attn3

        # aggregate

        out = einsum('... i j, ... j d -> ... i d', attn, v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, :n]

        if return_attn:
            return out, attn

        return out


def cycle(loader, cond=lambda x: True):
    while True:
        for data in loader:
            if not cond(data):
                continue
            yield data


class SampleHackDataset(Dataset):
    def __init__(self, meta, msa, seq, sst) -> None:
        super(SampleHackDataset, self).__init__()
        # TODO
        """
        1. load a meta df with all the ids and meta
        2. __getitem__(n) read n-th (seq, msa), sst from filesystem
        3. __dict__(key) read id -> (seq, msa), sst
        """
        self.meta = pd.read_csv(meta, index_col=0).reset_index(drop=True)
        self.msa_root = msa
        self.seq_root = seq
        self.sst_root = sst

    def __getitem__(self, index):
        item = self.meta.iloc[index, :]
        return self.read_from_disk_return_x_y(item)

    def read_from_disk_return_x_y(self, item):
        id = item["id"]
        seq_len = item["seq_len"]
        msa_depth = item["msa_depth"]
        with open(self.seq_root / f"{id}.seq", "rb") as f:
            seq = np.load(f)
            f.close()
        with open(self.msa_root / f"{id}.msa", "rb") as f:
            msa = np.load(f)
            f.close()
        with open(self.sst_root / f"{id}.sst", "rb") as f:
            sst = np.load(f)
            f.close()

        #
        return id, seq_len, msa_depth, seq, msa, sst

    def __len__(self):
        return self.meta.shape[0]

    def query(self, *args):
        return self.meta.query(*args)


class SecondaryAttention(NystromAttention):
    """ v0.1 attention
    v0.2 sparseattention
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        dim_head=64,
        heads=8,
        m=256,
        pinv_iterations=6,
        residual=True,
        eps=1e-8,
        ff_dropout=0.,
    ):
        super(SecondaryAttention, self).__init__(in_dim, out_dim, dim_head, heads, m, pinv_iterations, residual, eps)

        self.dropout = nn.Dropout(ff_dropout)
        self.in_norm = nn.LayerNorm(in_dim)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.m, self.pinv_iterations, self.eps

        x = self.in_norm(x)

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, 0, padding), value=0)

            if exists(mask):
                mask = F.pad(mask, (0, padding), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: self.in_norm(t), (q, k, v))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q *= self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        attn = attn1 @ attn2_inv @ attn3

        # aggregate

        out = einsum('... i j, ... j d -> ... i d', attn, v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, :n]

        if return_attn:
            return out, attn

        return out


class SimpleHackAttention(nn.Module):
    def __init__(self, num_class, in_dim=256, heads=8, head_dim=64, m=256, pinv_iterations=6, eps=1e-8):
        super().__init__()
        self.m = m
        self.heads = heads
        self.head_dim = head_dim
        self.pinv_iterations = pinv_iterations
        self.eps = eps
        self.num_class = num_class
        inner_dim = heads * head_dim
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, num_class)

    def forward(self, x, return_attn=False):
        b, n, d, h, m, iters, eps = *x.shape, self.heads, self.m, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, 0, padding), value=0)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        q *= self.scale
        # generate landmarks by sum reduction, and then calculate mean using the mask
        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)
        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean
        divisor = l

        # mean

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)
        # eq(15) from the paper
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        attn = attn1 @ attn2_inv @ attn3

        # aggregate
        out = einsum('... i j, ... j d -> ... i d', attn, v)

        # merge and combine
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, :n]

        return out, attn if return_attn else out


class SSModule(nn.Module):
    def __init__(self, num_class, **kwargs):
        super().__init__()
        # ff_dropout = kwargs.get('ff_dropout', 0.)
        # prenorm_256 = partial(PreNorm, 256)
        # prenorm_64 = partial(PreNorm, 64)
        # prenorm_8 = partial(PreNorm, 8)

        # self.q1 = SecondaryAttention(in_dim=256, out_dim=64, heads=8, dim_head=8, m=32, ff_dropout=ff_dropout)
        # self.ff1 = FeedForward(dim=64, dropout=ff_dropout)
        # self.q2 = SecondaryAttention(in_dim=64, out_dim=8, heads=4, dim_head=2, ff_dropout=ff_dropout)
        # self.ff2 = FeedForward(dim=8, dropout=ff_dropout)

        # self.out = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, num_class))
        self.sha = SimpleHackAttention(num_class, **kwargs)

    def forward(self, x):
        # # print(f'pre_ssp_forward: x.shape -> {x.shape}')
        # x = self.q1(x)
        # # print(f'post_ssp_q1: x.shape -> {x.shape}')
        # x = self.ff1(x)
        # # print(f'post_ssp_ff1: x.shape -> {x.shape}')
        # x = self.q2(x)
        # # print(f'post_ssp_q2: x.shape -> {x.shape}')
        # x = self.ff2(x)
        # # print(f'post_ssp_ff1: x.shape -> {x.shape}')
        # x = self.out(x)
        # # print(f'post_ssp_out: x.shape -> {x.shape}')
        x = self.sha(x)
        return x


def rand_choice(orginal_size: int, target_size: int, container):
    idxs = np.random.choice(orginal_size, target_size)
    container.extend(idxs)
    return idxs


def rand_chunk(max_depth, size, container):
    start = np.random.randint(low=0, high=max_depth - size - 1)
    res = range(start, start + size)
    container.extend([start])
    return res


def test(root: str):
    root = Path(root)
    device = torch.device('cuda:0')
    ds = SampleHackDataset(root / "sample.csv", root / "msa", root / "seq", root / "sst")
    af2 = Alphafold2(
        dim=256,
        depth=6,
        heads=8,
        dim_head=64,
        num_tokens=constants.NUM_AMINO_ACIDS_EXP,
        sparse_self_attn=True,
        cross_attn_compress_ratio=3,
        reversible=True,
    ).cuda()
    ssp = SSModule(num_class=NUM_SS_CLASS, in_dim=256).cuda()

    optim = torch.optim.Adam(list(af2.parameters()) + list(ssp.parameters()), lr=LEARNING_RATE)
    lossFn = nn.CrossEntropyLoss()
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    dl = cycle(ds, lambda x: x[2] > 1)

    step = 0
    global_counter = 0
    gradient_counter = 0

    def accumulated_enough(acc_size, curr):
        if curr >= acc_size:
            return True
        return False

    while True:
        step += 1
        id, seq_len, msa_depth, seq, msa, sst = next(dl)
        # preprocessing of seq, msa and sst
        # print(id, seq_len, msa_depth, seq.shape, msa.shape, sst.shape)

        # need seq to be shape of (1, l), msa of (1, l, d), sst of (1, l)

        seq = torch.tensor([int(AA[x]) for x in seq], dtype=torch.long)
        msa = torch.tensor([[int(AA[x]) for x in _seq] for _seq in msa], dtype=torch.long)
        sst = torch.tensor(sst, dtype=torch.long)
        print(f"id {id}: seq {seq.dtype}{seq.shape} msa {msa.dtype}{msa.shape} sst {sst.dtype}{sst.shape}")

        # seq = pad(seq, MAX_SEQ_LEN, b, seq_len)
        # msa = pad(msa, MAX_SEQ_LEN, b, seq_len, msa_depth=msa_depth)
        # sst = pad(sst, MAX_SEQ_LEN, b, seq_len)

        l = 128
        k = np.math.ceil(seq_len / l)
        r_k = k * l - seq_len
        id_extension_list = [f"{id}_k{ch_i}" for ch_i in range(k)]
        seq = F.pad(seq, (0, r_k))
        seq = torch.tensor([[chunk for chunk in seq[i * l:(i + 1) * l]] for i in range(k)])
        d = 4
        ml = 64
        msa = F.pad(msa, (0, r_k))
        msa_row_idxs = []
        msa_col_idxs = []
        msa = torch.tensor(
            [
                [
                    [chunk for chunk in _seq[rand_chunk(l, ml, msa_col_idxs)]]
                    for _seq in msa[rand_chunk(msa_depth, d, msa_row_idxs)]
                ] for _ in range(k)
            ]
        )
        sst = F.pad(sst, (0, r_k), value=-1)
        sst = torch.tensor([[chunk for chunk in sst[i * l:(i + 1) * l]] for i in range(k)])
        print(f"padded: seq {seq.shape} msa {msa.shape} sst {sst.shape}")
        # seq = rearrange(seq, 'b (limit chunk) -> (b chunk) limit', b = 1, limit = MAX_SEQ_LEN) # group result
        # msa = rearrange(msa, 'b (l_w w) (l_h h) -> (l_h b) (l_w b) h w', l_w = MAX_SEQ_LEN, l_h=MAX_NUM_MSA)
        # sst = rearrange(sst, 'b (l c) -> (b c) l', b = 1, l = MAX_SEQ_LEN)
        tloss = 0
        for i in range(k):
            n_seq, n_msa, cut_off = reshape_input(seq, msa, sst, i)
            trunk, *_ = af2(
                seq=n_seq, msa=n_msa, ss_only=True, msa_col_pos=msa_col_idxs[i], msa_row_pos=msa_row_idxs[i], seq_pos=i * l
            )
            ss = ssp(trunk)
            ss, ss_t = reshape_output(sst, i, cut_off, ss)
            loss = lossFn(ss, ss_t)
            tloss += loss

        write_loss(writer, global_counter, id, tloss)
        tloss.backward()
        global_counter += 1
        optim.step()
        optim.zero_grad()


def write_loss(writer, global_counter, id, tloss):
    writer.add_scalar(f"Loss v0.2", tloss, global_step=global_counter)
    writer.add_text("Data id", f"{id}", global_step=global_counter)


def reshape_output(sst, i, cut_off, ss):
    ss = rearrange(ss, 'b n c -> n b c', c=NUM_SS_CLASS)
    ss = ss[:cut_off, ...]
    ss = rearrange(ss, 'n b c -> b c n')
    ss_t = sst[i][:cut_off]
    ss_t = rearrange(ss_t, 'l -> () l').cuda()
    return ss, ss_t


def reshape_input(seq, msa, sst, i):
    n_seq = rearrange(seq[i], 'l -> () l').cuda()
    n_msa = rearrange(msa[i], 's w -> () s w').cuda()
    valid = sst[i][:] != -1
    cut_off = None
    for z in range(len(valid)):
        if valid[z]:
            cut_off = z
    print(f"pre_forward_shape: seq -> {n_seq.shape}, msa -> {n_msa.shape}")
    return n_seq, n_msa, cut_off


if __name__ == '__main__':
    test('./experiment/sample_msa')
