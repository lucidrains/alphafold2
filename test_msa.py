from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from torch.utils.data import Dataset
from inspect import isfunction
from functools import partial

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


# helper functions and classes
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(loader, cond=lambda x: True):
    while True:
        for data in loader:
            if not cond(data):
                continue
            yield data


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
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


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
        prot_id = item["id"]
        seq_len = item["seq_len"]
        msa_depth = item["msa_depth"]
        with open(self.seq_root / f"{id}.seq", "rb") as f:
            seq = np.load(f)
        with open(self.msa_root / f"{id}.msa", "rb") as f:
            msa = np.load(f)
        with open(self.sst_root / f"{id}.sst", "rb") as f:
            sst = np.load(f)

        return prot_id, seq_len, msa_depth, seq, msa, sst

    def __len__(self):
        return self.meta.shape[0]

    def query(self, *args):
        return self.meta.query(*args)


def rand_choice(orginal_size: int, target_size: int, container):
    idxs = np.random.choice(orginal_size, target_size)
    container.extend(idxs)
    return idxs


def rand_chunk(max_depth, size, container):
    start = np.random.randint(low=0, high=max_depth - size - 1)
    res = range(start, start + size)
    container.extend([start])
    return res


class TransPorter(nn.Module):
    def __init__(self):
        super().__init__()


def test(root: str):
    root = Path(root)
    device = torch.device('cuda:0')
    ds = SampleHackDataset(root / "sample.csv", root / "msa", root / "seq", root / "sst")

    tp = TransPorter(num_class=NUM_SS_CLASS, in_dim=256).cuda()

    optim = torch.optim.Adam(tp.parameters(), lr=LEARNING_RATE)
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
            ss, *_ = tp(
                seq=n_seq, msa=n_msa, ss_only=True, msa_col_pos=msa_col_idxs[i], msa_row_pos=msa_row_idxs[i], seq_pos=i * l
            )
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
