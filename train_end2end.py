import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange

import sidechainnet as scn
from alphafold2_pytorch import Alphafold2, DISTOGRAM_BUCKETS
from utils import *


# constants

DEVICE = None # defaults to cuda if available, else cpu
NUM_BATCHES = int(1e5)
GRADIENT_ACCUMULATE_EVERY = 16
LEARNING_RATE = 3e-4
IGNORE_INDEX = -100
THRESHOLD_LENGTH = 250

# set device

if DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(DEVICE)

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
    dynamic_batching = False,
    return_masks = True
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
dispersion_weight = 0.1
criterion = nn.MSELoss()
optim = Adam(model.parameters(), lr = LEARNING_RATE)

# training loop

for _ in range(NUM_BATCHES):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        _, seq, _, mask, *_, coords = next(dl)
        b, l = seq.shape

        # prepare mask, labels

        seq, coords, mask = seq.to(DEVICE), coords.to(DEVICE), mask.to(DEVICE).bool()
        coords = rearrange(coords, 'b (l c) d -> b l c d', l = l)

        # predict

        distogram = model(seq, mask = mask)

        # convert to 3d
        N_mask, CA_mask = scn_seq_mask(seq)
        distances, dispersion, weights = center_distogram_torch(distogram)

        coords_3d = MDScaling(distances, 
            weights,
            iters = 200, 
            fix_mirror = 5, 
            N_mask = N_mask,
            CA_mask = CA_mask
        ) 

        # refine

        # TODO: coords_align = refiner(coords_align)

        # rotate / align

        coords_aligned = Kabsch(coords_3d, coords)

        # loss
        loss = torch.sqrt(criterion(coords_aligned, coords)) + dispersion_weight * torch.norm(dispersion)

        loss.backward()

    print('loss:', loss.item())

    optim.step()
    optim.zero_grad()
