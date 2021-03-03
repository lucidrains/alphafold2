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

FEATURES = "esm" # one of ["esm", "msa", None]
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

def get_esm_embedd(seq):
    str_seq = "".join([VOCAB._int2char[x]for x in seq.cpu().numpy()])
    batch_labels, batch_strs, batch_tokens = batch_converter( [(0, str_seq)] )
    with torch.no_grad():
        results = embedd_model(batch_tokens, repr_layers=[33], return_contacts=False)
    return results["representations"][33].to(DEVICE)

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
    pos_tokens = 3, # N-term, C-alpha, C-term
    depth = 1,
    heads = 8,
    dim_head = 64
).to(DEVICE)

refiner = SE3Transformer(
    num_tokens = 10, # 10 unique atoms ([N-term, C-alpha, C-term, C-beta, =O, -O], C, O, N, S )
    dim = 64,
    depth = 2,
    input_degrees = 1,
    num_degrees = 2,
    output_degrees = 2,
    reduce_dim_out = True
)

# optimizer 

dispersion_weight = 0.1
criterion = nn.MSELoss()
optim = Adam(model.parameters(), lr = LEARNING_RATE)

# training loop

for _ in range(NUM_BATCHES):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        _, seq, _, mask, *_, coords = next(dl)
        b, l = seq.shape

        # prepare data and mask labels

        seq, coords, mask = seq.to(DEVICE), coords.to(DEVICE), mask.to(DEVICE)
        # coords = rearrange(coords, 'b (l c) d -> b l c d', l = l) # no need to rearrange for now
        # mask the atoms and backbone positions for each residue
        N_mask, CA_mask = scn_backbone_mask(seq, boolean = True, l_aa = 3) # NUM_COORDS_PER_RES
        cloud_mask = scn_cloud_mask(seq, boolean = False, current_mask = mask)
        # flatten last dims for point cloud masking and chain masking (cloud and sidechainnet). 
        chain_mask = (mask * cloud_mask)
        cloud_mask = rearrange(cloud_mask, 'b l c -> b (l c)', l = l).bool()
        chain_mask = rearrange(chain_mask, 'b l c -> b (l c)', l = l).bool()
        chain_mask = chain_mask[cloud_mask]

        # sequence embedding (msa / esm / attn / or nothing)
        msa, embedds = None
        # get embedds
        if FEATURES == "esm":
            embedds = get_esm_embedd(seq)
        # get msa here
        elif FEATURES == "msa":
            pass 
        # no embeddings 
        else:
            pass

        # elongate inputs by a factor of 3 : N-term, C-alpha C-term
        back = 4
        seq  = repeat(seq, 'b l -> b l back', back = back)
        seq  = rearrange(seq, 'b l back -> b (l back)')
        seq_pos = repeat(torch.arange(seq.shape[-1]) % back, 'lback -> b lback', b=seq.shape[0])
        mask = repeat(mask, 'b l -> b l back', back = back)
        mask = rearrange(mask, 'b l back -> b (l back)')
        if FEATURES == "msa":
            msa = repeat(msa, 'b l d -> b l back d', back = back)
            msa = rearrange(msa, 'b l back d -> b (l back) d')
        if FEATURES == "esm": 
            embedds = repeat(embedds, 'b l d -> b l back d', back = back)
            embedds = rearrange(embedds, 'b l back d -> b (l back) d')

        # predict - out is (batch, L*3, 3)
        distogram = model([seq, seq_pos], msa = msa, embedds = embedds, mask = mask) 

        # convert to 3d (batch, N, N, d) -> (batch, N, N) -> (batch, N, 3)
        distances, weights = center_distogram_torch(distogram)

        coords_3d, stress = MDScaling(distances, 
            weights,
            iters = 200, 
            fix_mirror = 5, 
            N_mask = N_mask,
            CA_mask = CA_mask
        ) 

        # build SC container. set SC points to CA and optionally place carbonyl O
        proto_sidechain = sidechain_container(seq, coords_3d, place_oxygen=False)
        proto_sidechain = rearrange(sidechain_3d, 'b l c d -> b (l c) d')
        
        ## refine
        # sample tokens for now based on indices
        atom_tokens = repeat(torch.arange(cloud_mask.shape[-1]), 'l -> b l', b=seq.shape[0]) % NUM_COORDS_PER_RES
        refined = refiner(atom_tokens[cloud_mask], proto_sidechain[cloud_mask], mask=chain_mask, return_type=1) # (batch, N, 3)

        # rotate / align
        coords_aligned, labels_aligned = Kabsch(refined, coords[cloud_mask])

        # save pdb files for visualization
        if TO_PDB: 
            # idx from batch to save prot and label
            idx = 0
            # create wrappers to 0
            wrapper_pred = rearrange( torch.zeros_like( coords[idx] ), '(l c) d -> l c d')
            wrapper_target = wrapper_pred.clone()
            wrapper_pred[cloud_mask] = coords_aligned[idx]
            wrapper_target[cloud_mask] = labels_aligned[idx]
            # build structures and save
            sb_pred = scn.StructureBuilder( seq[idx, :, 0], crd=wrapper_pred ) 
            sb_target = scn.StructureBuilder( seq[idx, :, 0], crd=wrapper_target ) 
            sb_pred.to_pdb(SAVE_DIR+"pred.pdb")
            sb_target.to_pdb(SAVE_DIR+"target.pdb")

        # loss - RMSE + distogram_dispersion
        loss = torch.sqrt(criterion(coords_aligned[chain_mask], labels_aligned[chain_mask])) + \
               dispersion_weight * torch.norm( (1/weights)-1 )

        loss.backward()

    print('loss:', loss.item())

    optim.step()
    optim.zero_grad()
