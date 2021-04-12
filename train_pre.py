import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange

import sidechainnet as scn
from alphafold2_pytorch import Alphafold2
import alphafold2_pytorch.constants as constants
from alphafold2_pytorch.utils import get_bucketed_distance_matrix

# constants

DEVICE = None # defaults to cuda if available, else cpu
NUM_BATCHES = int(1e5)
GRADIENT_ACCUMULATE_EVERY = 16
LEARNING_RATE = 3e-4
IGNORE_INDEX = -100
THRESHOLD_LENGTH = 250

# set device

DISTOGRAM_BUCKETS = constants.DISTOGRAM_BUCKETS
DEVICE = constants.DEVICE

# helpers

def cycle(loader, cond = lambda x: True):
    while True:
        for data in loader:
            if not cond(data):
                continue
            yield data

# get data

data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = 1,
    dynamic_batching = False
)

data = iter(data['train'])
data_cond = lambda t: t[1].shape[1] < THRESHOLD_LENGTH
dl = cycle(data, data_cond)

# model

model = Alphafold2(
    dim = 256,
    depth = 1,
    heads = 8,
    dim_head = 64
).to(DEVICE)

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

# training loop

for _ in range(NUM_BATCHES):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(dl)
        seq, coords, mask = batch.seqs, batch.crds, batch.msks

        b, l, _ = seq.shape

        # prepare mask, labels

        seq, coords, mask = seq.argmax(dim = -1).to(DEVICE), coords.to(DEVICE), mask.to(DEVICE).bool()
        coords = rearrange(coords, 'b (l c) d -> b l c d', l = l)

        discretized_distances = get_bucketed_distance_matrix(coords[:, :, 1], mask, DISTOGRAM_BUCKETS, IGNORE_INDEX)

        # predict

        distogram = model(seq, mask = mask)
        distogram = rearrange(distogram, 'b i j c -> b c i j')

        # loss

        loss = F.cross_entropy(
            distogram,
            discretized_distances,
            ignore_index = IGNORE_INDEX
        )

        loss.backward()

    print('loss:', loss.item())

    optim.step()
    optim.zero_grad()
