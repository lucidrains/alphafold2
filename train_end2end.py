import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange

# data

import sidechainnet as scn
from sidechainnet.sequence.utils import VOCAB
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES

# models

from alphafold2_pytorch import Alphafold2
import alphafold2_pytorch.constants as constants

from se3_transformer_pytorch import SE3Transformer
from alphafold2_pytorch.utils import *


# constants

FEATURES = "esm" # one of ["esm", "msa", "msa_transformer", None]
DEVICE = None # defaults to cuda if available, else cpu
NUM_BATCHES = int(1e5)
GRADIENT_ACCUMULATE_EVERY = 16
LEARNING_RATE = 3e-4
IGNORE_INDEX = -100
THRESHOLD_LENGTH = 250
TO_PDB = False
SAVE_DIR = ""

# set device

DEVICE = constants.DEVICE
DISTOGRAM_BUCKETS = constants.DISTOGRAM_BUCKETS

# set emebdder model from esm if appropiate - Load ESM-1b model

if FEATURES == "esm":
    # from pytorch hub (almost 30gb)
    embedd_model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
    batch_converter = alphabet.get_batch_converter()
    ##  alternatively do
    # import esm # after installing esm
    # model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()



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
    dim_head = 64,
    predict_coords = True,
    structure_module_dim = 8,
    structure_module_depth = 2,
    structure_module_heads = 4,
    structure_module_dim_head = 16,
    structure_module_refinement_iters = 2
).to(DEVICE)

# optimizer 

dispersion_weight = 0.1
criterion = nn.MSELoss()
optim = Adam(model.parameters(), lr = LEARNING_RATE)

# training loop

for _ in range(NUM_BATCHES):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(dl)
        seq, coords, mask = batch.seqs, batch.crds, batch.msks

        b, l, _ = seq.shape

        # prepare data and mask labels
        seq, coords, mask = seq.argmax(dim = -1).to(DEVICE), coords.to(DEVICE), mask.to(DEVICE)
        # coords = rearrange(coords, 'b (l c) d -> b l c d', l = l) # no need to rearrange for now
        # mask the atoms and backbone positions for each residue

        # sequence embedding (msa / esm / attn / or nothing)
        msa, embedds = None

        # get embedds
        if FEATURES == "esm":
            embedds = get_esm_embedd(seq, embedd_model, batch_converter)
        # get msa here
        elif FEATURES == "msa":
            pass 
        # no embeddings 
        else:
            pass

        # predict - out is (batch, L * 3, 3)

        refined = model(
            seq,
            msa = msa,
            embedds = embedds,
            mask = mask
        )

        # build SC container. set SC points to CA and optionally place carbonyl O
        proto_sidechain = sidechain_container(coords_3d, n_aa=batch,
                                              cloud_mask=cloud_mask, place_oxygen=False)

        # rotate / align
        coords_aligned, labels_aligned = Kabsch(refined, coords[flat_cloud_mask])

        # atom mask

        cloud_mask = scn_cloud_mask(seq, boolean = False)
        flat_cloud_mask = rearrange(cloud_mask, 'b l c -> b (l c)')

        # chain_mask is all atoms that will be backpropped thru -> existing + trainable 

        chain_mask = (mask * cloud_mask)[cloud_mask]
        flat_chain_mask = rearrange(chain_mask, 'b l c -> b (l c)')

        # save pdb files for visualization

        if TO_PDB: 
            # idx from batch to save prot and label
            idx = 0
            coords2pdb(seq[idx, :, 0], coords_aligned[idx], cloud_mask, prefix=SAVE_DIR, name="pred.pdb")
            coords2pdb(seq[idx, :, 0], labels_aligned[idx], cloud_mask, prefix=SAVE_DIR, name="label.pdb")

        # loss - RMSE + distogram_dispersion
        loss = torch.sqrt(criterion(coords_aligned[flat_chain_mask], labels_aligned[flat_chain_mask])) + \
                          dispersion_weight * torch.norm( (1/weights)-1 )

        loss.backward()

    print('loss:', loss.item())

    optim.step()
    optim.zero_grad()
