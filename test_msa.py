from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch.utils.data import Dataset

from alphafold2_pytorch import Alphafold2, Attention

# n: 20171106 : changed-to: '.' = 0
AA = {
    '.': 0,
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

NUM_BATCHES = int(1e5)
GRADIENT_ACCUMULATE_EVERY = 16
LEARNING_RATE = 3e-5
IGNORE_INDEX = -100
THRESHOLD_LENGTH = 250
EMBEDDING_SIZE = np.max(list(AA.values())) + 1  # 22
MAX_SEQ_LEN = 2048
MAX_NUM_MSA = 8192
BATCH_SIZE = 1
NUM_SS_CLASS = 3
DISTANCE_BINS = 37


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

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()


class SSWrapper(nn.Module):
    """due to the nature of available msa->sst training data, forward need to handle per amino acid position logic
    thus we create a wrapper largely duplicate alphafold2's forward but with modification
    """
    def __init__(self, num_q, **kwargs):
        super().__init__()
        dim = kwargs.pop('dim', None)
        self.af2 = Alphafold2(**kwargs)
        self.to_secondary_structure = Attention()


def test(root: str):
    root = Path(root)
    device = torch.device('cuda:0')
    ds = SampleHackDataset(root / "meta.csv", root / "msa", root / "seq", root / "sst")
    model = SSWrapper(
        num_q=3,
        dim=256,
        depth=6,
        max_seq_len=MAX_SEQ_LEN,
        embedding_size=EMBEDDING_SIZE,
        max_msa_depth=MAX_NUM_MSA,
        heads=8,
        dim_head=64
    ).cuda()

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lossFn = nn.CrossEntropyLoss()
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    dl = cycle(ds, lambda x: x[2] > 1)

    step = 0
    global_counter = 0
    b = BATCH_SIZE
    gradient_counter = 0

    def accumulated_enought(acc_size, curr):
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

        l = 96
        k = np.math.ceil(seq_len / l)
        r_k = k * l - seq_len
        id_extension_list = [f"{id}_w{ch_i}" for ch_i in range(k)]
        seq = torch.nn.functional.pad(seq, (0, r_k))
        seq = torch.tensor([[chunk for chunk in seq[i * l:(i + 1) * l]] for i in range(k)])
        d = 8
        j = np.math.ceil(msa_depth / d)
        r_j = d * j - msa_depth
        msa = torch.nn.functional.pad(msa, (0, r_k, 0, r_j))
        msa = torch.tensor(
            [[[[ch for ch in _seq[i * l:(i + 1) * l]] for i in range(k)] for _seq in msa[h * d:(h + 1) * d]] for h in range(j)]
        )
        msa = rearrange(msa, 'h d b w -> b h d w')
        sst = torch.nn.functional.pad(sst, (0, r_k), value=-1)
        sst = torch.tensor([[chunk for chunk in sst[i * l:(i + 1) * l]] for i in range(k)])
        print(f"padded: seq {seq.shape} msa {msa.shape} sst {sst.shape}")
        # seq = rearrange(seq, 'b (limit chunk) -> (b chunk) limit', b = 1, limit = MAX_SEQ_LEN) # group result
        # msa = rearrange(msa, 'b (l_w w) (l_h h) -> (l_h b) (l_w b) h w', l_w = MAX_SEQ_LEN, l_h=MAX_NUM_MSA)
        # sst = rearrange(sst, 'b (l c) -> (b c) l', b = 1, l = MAX_SEQ_LEN)
        i, j = 0, 0

        for b in msa:
            for h in b:
                n_seq = rearrange(seq[i], 'l -> () l').cuda()
                n_msa = rearrange(h, 'd h -> () d h').cuda()

                valid = sst[i][:] != -1
                cut_off = None
                for z in range(len(valid)):
                    if valid[z]:
                        cut_off = z

                ss = model(seq=n_seq, w_pos=i * l, msa=n_msa, h_pos=j * d)

                ss = rearrange(ss, 'b n c -> b c n', c=NUM_SS_CLASS).cpu()
                ss = rearrange(ss, 'b c n -> n b c')
                ss = ss[:cut_off, ...]
                ss = rearrange(ss, 'n b c -> b c n')
                ss_t = sst[i][:cut_off]
                ss_t = rearrange(ss_t, 'l -> () l')

                loss = lossFn(ss, ss_t)
                gradient_counter += 1
                writer.add_scalar(
                    f"Loss/train per {GRADIENT_ACCUMULATE_EVERY} miniblocks in collated samples", loss, global_counter
                )
                global_counter += 1

                loss.backward()
                if accumulated_enought(GRADIENT_ACCUMULATE_EVERY, gradient_counter):
                    gradient_counter = 0
                    optim.step()
                    optim.zero_grad()

                j += 1
            i += 1
            # optim.step()
            # optim.zero_grad()
        # ss = model(seq=seq, msa=msa)

        # # print(f"model result: {ss.dtype} {ss.shape}")

        # ss = rearrange(ss, 'b n c -> b c n', c=NUM_SS_CLASS)
        # loss = lossFn(ss, sst)
        # writer.add_scalar("Loss/train - per sequence", loss, step)
        # loss.backward()

        # del id, seq_len, msa_depth, seq, msa, sst

        # optim.step()
        # optim.zero_grad()

        # writer.flush()
    writer.close()


if __name__ == '__main__':
    test()
