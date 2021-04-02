# utils for working with 3d-protein structures
import os
import numpy as np
import torch
from functools import wraps
from einops import rearrange, repeat

# sidechainnet

from sidechainnet.utils.sequence import ProteinVocabulary, ONE_TO_THREE_LETTER_MAP
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, BB_BUILD_INFO, SC_BUILD_INFO
from sidechainnet.structure.StructureBuilder import _get_residue_build_iter

# build vocabulary

VOCAB = ProteinVocabulary()

# constants

import alphafold2_pytorch.constants as constants

# constants: same as in alphafold2.py

DISTANCE_THRESHOLDS = torch.linspace(2, 20, steps = constants.DISTOGRAM_BUCKETS)

# distance binning function

def get_bucketed_distance_matrix(coords, mask, num_buckets = constants.DISTOGRAM_BUCKETS, ignore_index = -100):
    distances = torch.cdist(coords, coords, p=2)
    boundaries = torch.linspace(2, 20, steps = num_buckets, device = coords.device)
    discretized_distances = torch.bucketize(distances, boundaries[:-1])
    discretized_distances.masked_fill_(~(mask[..., None] & mask[..., None, :]), ignore_index)
    return discretized_distances

# decorators

def set_backend_kwarg(fn):
    @wraps(fn)
    def inner(*args, backend = 'auto', **kwargs):
        if backend == 'auto':
            backend = 'torch' if isinstance(args[0], torch.Tensor) else 'numpy'
        kwargs.update(backend = backend)
        return fn(*args, **kwargs)
    return inner

def expand_dims_to(t, length = 3):
    if length == 0:
        return t
    return t.reshape(*((1,) * length), *t.shape) # will work with both torch and numpy

def expand_arg_dims(dim_len = 3):
    """ pack here for reuse. 
        turns input into (B x D x N)
    """
    def outer(fn):
        @wraps(fn)
        def inner(x, y, **kwargs):
            assert len(x.shape) == len(y.shape), "Shapes of A and B must match."
            remaining_len = dim_len - len(x.shape)
            x = expand_dims_to(x, length = remaining_len)
            y = expand_dims_to(y, length = remaining_len)
            return fn(x, y, **kwargs)
        return inner
    return outer

def invoke_torch_or_numpy(torch_fn, numpy_fn):
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            backend = kwargs.pop('backend')
            passed_args = fn(*args, **kwargs)
            passed_args = list(passed_args)
            if isinstance(passed_args[-1], dict):
                passed_kwargs = passed_args.pop()
            else:
                passed_kwargs = {}
            backend_fn = torch_fn if backend == 'torch' else numpy_fn
            return backend_fn(*passed_args, **passed_kwargs)
        return inner
    return outer


# preprocess data

def get_atom_ids_dict():
    """ Get's a dict mapping each atom to a token. """
    ids = set(["", "N", "CA", "C", "O"])

    for k,v in SC_BUILD_INFO.items():
        for name in v["atom-names"]:
            ids.add(name)
            
    return {k: i for i,k in enumerate(sorted(ids))}

def make_cloud_mask(aa):
    """ relevent points will be 1. paddings will be 0. """
    mask = np.zeros(14)
    # early stop if padding token
    if aa == "_":
        return mask
    # get num of atoms in aa
    n_atoms = 4+len( SC_BUILD_INFO[ ONE_TO_THREE_LETTER_MAP[aa] ]["atom-names"] )
    mask[:n_atoms] = 1
    return mask

def make_atom_id_embedds(aa, atom_ids):
    """ Return the tokens for each atom in the aa. """
    mask = np.zeros(14)
    # early stop if padding token
    if aa == "_":
        return mask
    # get atom id
    atom_list = ["N", "CA", "C", "O"] + SC_BUILD_INFO[ ONE_TO_THREE_LETTER_MAP[aa] ]["atom-names"]
    for i,atom in enumerate(atom_list):
        mask[i] = ATOM_IDS[atom]
    return mask


ATOM_IDS = get_atom_ids_dict()
CUSTOM_INFO = {k: {"cloud_mask": make_cloud_mask(k),
                   "atom_id_embedd": make_atom_id_embedds(k, atom_ids=ATOM_IDS),
                  } for k in "ARNDCQEGHILKMFPSTWYV_"}

# common utils

# parsing to pdb for easier visualization - other example from sidechainnet is:
# https://github.com/jonathanking/sidechainnet/tree/master/sidechainnet/structure

def download_pdb(name, route):
    """ Downloads a PDB entry from the RCSB PDB. 
        Inputs:
        * name: str. the PDB entry id. 4 characters, capitalized.
        * route: str. route of the destin file. usually ".pdb" extension
        Output: route of destin file
    """
    os.system(f"curl https://files.rcsb.org/download/{name}.pdb > {route}")
    return route

def clean_pdb(name, route=None, chain_num=None):
    """ Cleans the structure to only leave the important part.
        Inputs: 
        * name: str. route of the input .pdb file
        * route: str. route of the output. will overwrite input if not provided
        * chain_num: int. index of chain to select (1-indexed as pdb files)
        Output: route of destin file.
    """
    import mdtraj
    destin = route if route is not None else name
    # read input
    raw_prot = mdtraj.load_pdb(name)
    # iterate over prot and select the specified chains
    idxs = []
    for chain in raw_prot.topology.chains:
        # if arg passed, only select that chain
        if chain_num is not None:
            if chain_num != chain.index:
                continue
        # select indexes of chain
        chain_idxs = raw_prot.topology.select(f"chainid == {str(chain.index)}")
        idxs.extend( chain_idxs.tolist() )
    # sort: topology and xyz selection are ordered
    idxs = sorted(idxs)
    # get new trajectory from the sleected subset of indexes and save
    prot = mdtraj.Trajectory(xyz=raw_prot.xyz[:, idxs], 
                             topology=raw_prot.topology.subset(idxs))
    prot.save(destin)
    return destin

def custom2pdb(coords, proteinnet_id, route):
    """ Takes a custom representation and turns into a .pdb file. 
        Inputs:
        * coords: array/tensor of shape (3 x N) or (N x 3). in Angstroms.
                  same order as in the proteinnnet is assumed (same as raw pdb file)
        * proteinnet_id: str. proteinnet id format (<class>#<pdb_id>_<chain_number>_<chain_id>)
                         see: https://github.com/aqlaboratory/proteinnet/
        * route: str. destin route.
        Output: tuple of routes: (original, generated) for the structures. 
    """
    import mdtraj
    # convert to numpy
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    # ensure (1, N, 3)
    if coords.shape[1] == 3:
        coords = coords.T
    coords = np.newaxis(coords, axis=0)
    # get pdb id and chain num
    pdb_name, chain_num = proteinnet_id.split("#")[-1].split("_")[:-1]
    pdb_destin = "/".join(route.split("/")[:-1])+"/"+pdb_name+".pdb"
    # download pdb file and select appropiate 
    download_pdb(pdb_name, pdb_destin)
    clean_pdb(pdb_destin, chain_num=chain_num)
    # load trajectory scaffold and replace coordinates - assumes same order
    scaffold = mdtraj.load_pdb(pdb_destin)
    scaffold.xyz = coords
    scaffold.save(route)
    return pdb_destin, route


def coords2pdb(seq, coords, cloud_mask, prefix="", name="af2_struct.pdb"):
    """ Turns coordinates into PDB files ready to be visualized. 
        Inputs:
        * seq: (L,) tensor of ints (sidechainnet aa-key pairs)
        * coords: (3, N) coords of atoms
        * cloud_mask: (L, C) boolean mask of occupied spaces in scn format
        * prefix: str. directory to save files.
        * name: str. name of destin file (ex: pred1.pdb)
    """
    scaffold = torch.zeros( cloud_mask.shape, 3 )
    scaffold[cloud_mask] = coords.cpu().float()
    # build structures and save
    pred = scn.StructureBuilder( seq, crd=scaffold ) 
    pred.to_pdb(prefix+name)


# sidechainnet / other data utils

def get_esm_embedd(seq, embedd_model, batch_converter, embedd_type="per_tok"):
    """ Returns the ESM embeddings for a protein. 
        Inputs: 
        * seq: (L,) tensor of ints (in sidechainnet int-char convention)
        * embedd_model: ESM model (see train_end2end.py for an example)
        * batch_converter: ESM batch converter (see train_end2end.py for an example)
        * embedd_type: one of ["mean", "per_tok"]. 
                       "per_tok" is recommended if working with sequences.
    """
    str_seq = "".join([VOCAB._int2char[x]for x in seq.cpu().numpy()])
    batch_labels, batch_strs, batch_tokens = batch_converter( [(0, str_seq)] )
    with torch.no_grad():
        results = embedd_model(batch_tokens, repr_layers=[33], return_contacts=False)
    # index 0 is for start token. so take from 1 one
    token_reps = results["representations"][33][ 1 : len(str_seq) + 1].to(seq.device)
    if embedd_type == "mean":
        token_reps = token_reps.mean(dim=0)
    return token_reps



def get_all_protein_ids(dataloader, verbose=False):
    """ Given a sidechainnet dataloader for a CASP version, 
        Returns all the ids belonging to proteins.
        Inputs: 
        * dataloader: a sidechainnet dataloader for a CASP version
        Outputs: a set containing the ids for all protein entries. 
    """
    # store ids here
    ids = set([])
    # iterate for all batches
    for i,batch in tqdm(enumerate(dataloaders['train'])):
        # for breaking from 2 loops at once
        try:
            for i in range(batch.int_seqs.shape[0]):
                # check if all fragments are : 4_LETTER_PDB + NUM + CHAIN
                max_len_10 = len(batch.pids[i]) < 10 
                fragments  = [len(x) <= 4 for x in batch.pids[i].split("_")] 
                fragments_under_4 = sum(fragments) == len(fragments) # AND CONDITION
                # record id 
                if max_len_10 and fragments_under_4:
                    ids.add(batch.pids[i])
                else: 
                    if verbose:
                        print("skip:", batch.pids[i], "under 4", fragments)
        except StopIteration:
            break
    # returns set of ids
    return ids
    

def scn_cloud_mask(scn_seq, boolean=True, coords=None):
    """ Gets the boolean mask atom positions (not all aas have same atoms). 
        Inputs: 
        * scn_seq: (batch, length) sequence as provided by Sidechainnet package
        * boolean: whether to return as array of idxs or boolean values
        * coords: optional .(batch, lc, 3). sidechainnet coords.
                  returns the true mask (solves potential atoms that might not be provided)
        Outputs: (batch, length, NUM_COORDS_PER_RES) boolean mask 
    """

    scn_seq = expand_dims_to(scn_seq, 2 - len(scn_seq.shape))
    # early check for coords mask
    if coords is not None: 
        batch_mask = ( rearrange(coords, '... (l c) d -> ... l c d', c=14) == 0 ).sum(dim=-1) < coords.shape[-1]
        if boolean:
            return batch_mask.bool()
        else: 
            return batch_mask.nonzero()

    # do loop in cpu
    device = scn_seq.device
    batch_mask = []
    scn_seq = scn_seq.cpu()
    for i, seq in enumerate(scn_seq):
        # get masks for each prot (points for each aa)
        batch_mask.append( torch.tensor([CUSTOM_INFO[VOCAB.int2char(aa.item())]['cloud_mask'] \
                                         for aa in seq]).bool().to(device).unsqueeze(0) )
    # concat in last dim
    batch_mask = torch.cat(batch_mask, dim=0)
    # return mask (boolean or indexes)
    if boolean:
        return batch_mask.bool()
    else: 
        return batch_mask.nonzero()

    

def scn_backbone_mask(scn_seq, boolean=True, n_aa=3):
    """ Gets the boolean mask for N and CA positions. 
        Inputs: 
        * scn_seq: sequence(s) as provided by Sidechainnet package (int tensor/s)
        * n_aa: number of atoms in a backbone. (may include cbeta as 4th pos)
        * bool: whether to return as array of idxs or boolean values
        Outputs: (N_mask, CA_mask, C_mask)
    """
    wrapper = torch.zeros(*scn_seq.shape, n_aa)
    # N is the first atom in every AA. CA is the 2nd.
    wrapper[:, 0] = 1
    wrapper[:, 1] = 2
    wrapper[:, 2] = 3
    wrapper = rearrange(wrapper, '... l c -> ... (l c)')
    # find idxs
    N_mask  = wrapper == 1
    CA_mask = wrapper == 2
    C_mask  = wrapper == 3 
    if boolean:
        return N_mask, CA_mask, C_mask
    return torch.nonzero(N_mask), torch.nonzero(CA_mask), torch.nonzero(C_mask)

def scn_atom_embedd(scn_seq):
    """ Returns the token for each atom in the aa. 
        Inputs: 
        * scn_seq: sequence(s) as provided by Sidechainnet package (int tensor/s)
    """
    device = scn_seq.device
    batch_tokens = []
    # do loop in cpu
    scn_seq = scn_seq.cpu()
    for i,seq in enumerate(scn_seq):
        batch_tokens.append( torch.tensor([CUSTOM_INFO[VOCAB.int2char(aa.item())]["atom_id_embedd"] \
                                           for aa in seq]).long().to(device).unsqueeze(0) )
    batch_tokens = torch.cat(batch_tokens, dim=0)
    return batch_tokens


def nerf_torch(a, b, c, l, theta, chi):
    """ Custom Natural extension of Reference Frame. 
        Inputs:
        * a: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * b: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * c: (batch, 3) or (3,). point(s) of the plane, connected to d
        * theta: (batch,) or (float).  angle(s) between b-c-d
        * chi: (batch,) or float. dihedral angle(s) between the a-b-c and b-c-d planes
        Outputs: d (batch, 3) or (3,). the next point in the sequence, linked to c
    """
    # safety check
    if not ( (-np.pi <= theta) * (theta <= np.pi) ).all().item():
        raise ValueError(f"theta(s) must be in radians and in [-pi, pi]. theta(s) = {theta}")
    # calc vecs
    ba = b-a
    cb = c-b
    # calc rotation matrix. based on plane normals and normalized
    n_plane  = torch.cross(ba, cb, dim=-1)
    n_plane_ = torch.cross(n_plane, cb, dim=-1)
    rotate   = torch.stack([cb, n_plane_, n_plane], dim=-1)
    rotate  /= torch.norm(rotate, dim=-2, keepdim=True)
    # calc proto point, rotate
    d = torch.stack([-torch.cos(theta),
                      torch.sin(theta) * torch.cos(chi),
                      torch.sin(theta) * torch.sin(chi)], dim=-1).unsqueeze(-1)
    # extend base point, set length
    return c + l.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()

def sidechain_container(backbones, n_aa, cloud_mask=None, place_oxygen=False,
                        n_atoms=NUM_COORDS_PER_RES, padding=GLOBAL_PAD_CHAR):
    """ Gets a backbone of the protein, returns the whole coordinates
        with sidechains (same format as sidechainnet). Keeps differentiability.
        Inputs: 
        * backbones: (batch, L*3, 3): assume batch=1 (could be extended later).
                    Coords for (N-term, C-alpha, C-term) of every aa.
        * n_aa: int. number of points for each aa in the backbones.
        * cloud_mask: (batch, l, c). optional. cloud mask from scn_cloud_mask`.
                      returns point outside to 0. if passed, else c_alpha
        * place_oxygen: whether to claculate the oxygen of the
                        carbonyl group via NeRF
        * n_atoms: int. n of atom positions / atom. same as in sidechainnet: 14
        * padding: int. padding token. same as in sidechainnet: 0
        Outputs: whole coordinates of shape (batch, L, n_atoms, 3)
    """
    device = backbones.device
    batch, length = backbones.shape[0], backbones.shape[1] // n_aa
    # build scaffold from (N, CA, C, CB)
    new_coords = torch.zeros(batch, length, NUM_COORDS_PER_RES, 3).to(device)
    predicted  = rearrange(backbones, 'b (l back) d -> b l back d', l=length)
    # set backbone positions
    new_coords[:, :, :3] = predicted[:, :, :3]
    # set rest of positions to c_beta if present, else c_alpha
    if n_aa == 4:
        new_coords[:, :, 4:] = repeat(predicted[:, :, -1], 'b l d -> b l scn d', scn=10)
    else:
        new_coords[:, :, 4:] = repeat(new_coords[:, :, 1], 'b l d -> b l scn d', scn=10)
    if cloud_mask is not None:
        new_coords[torch.logical_not(cloud_mask)] = 0.
    # hard-calculate oxygen position of carbonyl group with parallel version of NERF
    if place_oxygen: 
        # build (=O) position of revery aa in each chain
        for s in range(batch):
            # dihedrals phi=f(c-1, n, ca, c) & psi=f(n, ca, c, n+1)
            # phi = get_dihedral_torch(*backbone[s, i*3 - 1 : i*3 + 3]) if i>0 else None
            psis = torch.tensor([ get_dihedral_torch(*backbones[s, i*3 + 0 : i*3 + 4] )if i < length-1 else np.pi*5/4 \
                                  for i in range(length) ])
            # the angle for placing oxygen is opposite to psi of current res.
            # psi not available for last one so pi/4 taken for now
            bond_lens  = repeat(torch.tensor(BB_BUILD_INFO["BONDLENS"]["c-o"]), ' -> b', b=length).to(psis.device)
            bond_angs  = repeat(torch.tensor(BB_BUILD_INFO["BONDANGS"]["ca-c-o"]), ' -> b', b=length).to(psis.device)
            correction = repeat(torch.tensor(-np.pi), ' -> b', b=length).to(psis.device) 
            new_coords[:, :, 3] = nerf_torch(new_coords[:, :, 0], 
                                             new_coords[:, :, 1], 
                                             new_coords[:, :, 2], 
                                             bond_lens, bond_angs, psis + correction)
    else: 
        # init oxygen to carbonyl
        new_coords[:, :, 3] = predicted[:, :, 2]

    return new_coords



# distance utils (distogram to dist mat + masking)

def center_distogram_torch(distogram, bins=DISTANCE_THRESHOLDS, min_t=1., center="mean", wide="std"):
    """ Returns the central estimate of a distogram. Median for now.
        Inputs:
        * distogram: (batch, N, N, B) where B is the number of buckets.
        * bins: (B,) containing the cutoffs for the different buckets
        * min_t: float. lower bound for distances.
        Outputs:
        * central: (batch, N, N)
        * dispersion: (batch, N, N)
        * weights: (batch, N, N)
    """
    shape, device = distogram.shape, distogram.device
    # threshold to weights and find mean value of each bin
    n_bins = ( bins - 0.5 * (bins[2] - bins[1]) ).to(device)
    n_bins[0]  = 1.5
    n_bins[-1] = 1.33*bins[-1] # above last threshold is ignored
    max_bin_allowed = torch.tensor(n_bins.shape[0]-1).to(device).long()
    # calculate measures of centrality and dispersion - 
    if center == "median":
        cum_dist = torch.cumsum(distogram, dim=-1)
        medium   = 0.5 * torch.ones(*cum_dist.shape[:-1], 1, device=device)
        central  = torch.searchsorted(cum_dist, medium).squeeze()
        central  = n_bins[ torch.min(central, max_bin_allowed) ]
    elif center == "mean":
        central  = (distogram * n_bins).sum(dim=-1)
    # create mask for last class - (IGNORE_INDEX)   
    mask = (central <= bins[-2].item()).float()
    # mask diagonal to 0 dist - don't do masked filling to avoid inplace errors
    diag_idxs = np.arange(shape[-2])
    central   = expand_dims_to(central, 3 - len(central.shape))
    central[:, diag_idxs, diag_idxs]  *= 0.
    # provide weights
    if wide == "var":
        dispersion = (distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1)
    elif wide == "std":
        dispersion = (distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1).sqrt()
    else:
        dispersion = torch.zeros_like(central, device=device)
    # rescale to 0-1. lower std / var  --> weight=1. set potential nan's to 0
    weights = mask / (1+dispersion)
    weights[weights != weights] = 0.
    return central, weights

# distance matrix to 3d coords: https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/manifold/_mds.py#L279

def mds_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper. 
        Assumes (for now) distogram is (N x N) and symmetric
        Outs: 
        * best_3d_coords: (batch x 3 x N)
        * historic_stresses: (batch x steps)
    """
    device, dtype = pre_dist_mat.device, pre_dist_mat.type()

    if weights is None:
        weights = torch.ones_like(pre_dist_mat)

    # ensure batched MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length = ( 3 - len(pre_dist_mat.shape) ))

    # start
    batch, N, _ = pre_dist_mat.shape
    diag_idxs = np.arange(N)
    his = []
    # init random coords
    best_stress = float("Inf") * torch.ones(batch, device = device).type(dtype)
    best_3d_coords = 2*torch.rand(batch, N, 3, device = device).type(dtype) - 1
    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        dist_mat = torch.cdist(best_3d_coords, best_3d_coords, p=2).clone()
        stress   = ( weights * (dist_mat - pre_dist_mat)**2 ).sum(dim=(-1,-2)) * 0.5
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[ dist_mat == 0 ] = 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio
        B[:, diag_idxs, diag_idxs] += ratio.sum(dim=-1)

        # update
        coords = (1. / N * torch.matmul(B, best_3d_coords))
        dis = torch.norm(coords, dim=(-1, -2))
        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (best_stress - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        best_stress = (stress / dis)
        his.append(best_stress)

    return torch.transpose(best_3d_coords, -1,-2), torch.cat(his)

def mds_numpy(pre_dist_mat, weights=None, iters=10, tol=1e-5, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper. 
        Assumes (for now) distrogram is (N x N) and symmetric
        Out:
        * best_3d_coords: (3 x N)
        * historic_stress 
    """
    if weights is None:
        weights = np.ones_like(pre_dist_mat)

    # ensure batched MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length = ( 3 - len(pre_dist_mat.shape) ))
    # start
    batch, N, _ = pre_dist_mat.shape
    his = []
    # init random coords
    best_stress = np.inf * np.ones(batch)
    best_3d_coords = 2*np.random.rand(batch, 3, N) - 1
    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        dist_mat = np.linalg.norm(best_3d_coords[:, :, :, None] - best_3d_coords[:, :, None, :], axis=-3)
        stress   = (( weights * (dist_mat - pre_dist_mat) )**2).sum(axis=(-1, -2)) * 0.5
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[dist_mat == 0] = 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio 
        B[:, np.arange(N), np.arange(N)] += ratio.sum(axis=-1)
        # update - double transpose. TODO: consider fix
        coords = (1. / N * np.matmul(best_3d_coords, B))
        dis = np.linalg.norm(coords, axis=(-1, -2))
        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (best_stress - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        best_stress = stress / dis
        his.append(best_stress)

    return best_3d_coords, np.array(his)

def get_dihedral_torch(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Can't use torch.dot bc it does not broadcast
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) ) 


def get_dihedral_numpy(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return np.arctan2( ( (np.linalg.norm(u2, axis=-1, keepdims=True) * u1) * np.cross(u2,u3, axis=-1)).sum(axis=-1),  
                       ( np.cross(u1,u2, axis=-1) * np.cross(u2, u3, axis=-1) ).sum(axis=-1) ) 

def calc_phis_torch(pred_coords, N_mask, CA_mask, C_mask=None,
                    prop=True, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * N_mask: (batch, N) boolean mask for N-term positions
        * CA_mask: (batch, N) boolean mask for C-alpha positions
        * C_mask: (batch, N) or None. boolean mask for C-alpha positions or
                    automatically calculate from N_mask and CA_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
        Note: use [0] since all prots in batch have same backbone
    """ 
    # detach gradients for angle calculation - mirror selection
    pred_coords_ = torch.transpose(pred_coords.detach(), -1 , -2).cpu()
    # ensure dims
    N_mask = expand_dims_to( N_mask, 2-len(N_mask.shape) )
    CA_mask = expand_dims_to( CA_mask, 2-len(CA_mask.shape) )
    if C_mask is not None: 
        C_mask = expand_dims_to( C_mask, 2-len(C_mask.shape) )
    else:
        C_mask = torch.logical_not(torch.logical_or(N_mask,CA_mask))
    # select points
    n_terms  = pred_coords_[:, N_mask[0].squeeze()]
    c_alphas = pred_coords_[:, CA_mask[0].squeeze()]
    c_terms  = pred_coords_[:, C_mask[0].squeeze()]
    # compute phis for every pritein in the batch
    phis = [get_dihedral_torch(c_terms[i, :-1],
                               n_terms[i,  1:],
                               c_alphas[i, 1:],
                               c_terms[i,  1:]) for i in range(pred_coords.shape[0])]

    # return percentage of lower than 0
    if prop: 
        return torch.tensor( [(x<0).float().mean().item() for x in phis] ) 
    return phis


def calc_phis_numpy(pred_coords, N_mask, CA_mask, C_mask=None,
                    prop=True, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * N_mask: (N, ) boolean mask for N-term positions
        * CA_mask: (N, ) boolean mask for C-alpha positions
        * C_mask: (N, ) or None. boolean mask for C-alpha positions or
                    automatically calculate from N_mask and CA_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
    """ 
    # detach gradients for angle calculation - mirror selection
    pred_coords_ = np.transpose(pred_coords, (0, 2, 1))
    n_terms  = pred_coords_[:, N_mask.squeeze()]
    c_alphas = pred_coords_[:, CA_mask.squeeze()]
    # select c_term auto if not passed
    if C_mask is not None: 
        c_terms = pred_coords_[:, C_mask]
    else:
        c_terms  = pred_coords_[:, (np.ones_like(N_mask)-N_mask-CA_mask).squeeze().astype(bool) ]
    # compute phis for every pritein in the batch
    phis = [get_dihedral_numpy(c_terms[i, :-1],
                               n_terms[i,  1:],
                               c_alphas[i, 1:],
                               c_terms[i,  1:]) for i in range(pred_coords.shape[0])]

    # return percentage of lower than 0
    if prop: 
        return np.array( [(x<0).mean() for x in phis] ) 
    return phis


# alignment by centering + rotation to compute optimal RMSD
# adapted from : https://github.com/charnley/rmsd/

def kabsch_torch(X, Y, cpu=True):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    device = X.device
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t()).detach()
    if cpu: 
        C = C.cpu()
    # Optimal rotation matrix via SVD - warning! W must be transposed
    V, S, W = torch.svd(C)
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W.t()).to(device)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_, Y_

def kabsch_numpy(X, Y):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    # center X and Y to the origin
    X_ = X - X.mean(axis=-1, keepdims=True)
    Y_ = Y - Y.mean(axis=-1, keepdims=True)
    # calculate convariance matrix (for each prot in the batch)
    C = np.dot(X_, Y_.transpose())
    # Optimal rotation matrix via SVD
    V, S, W = np.linalg.svd(C)
    # determinant sign for direction correction
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = np.dot(V, W)
    # calculate rotations
    X_ = np.dot(X_.T, U).T
    # return centered and aligned
    return X_, Y_


# metrics - more formulas here: http://predictioncenter.org/casp12/doc/help.html

def distmat_loss_torch(X, Y, p=2, q=2, distmat_mask=None):
    """ Calculates a loss on the distance matrix - no need to align structs.
        Inputs: 
        * X: (N, d) tensor. the predicted stricture 
        * Y: (N, d) tensor. the true structure
        * p: int. power for the distance calculation (2 for euclidean)
        * q: float. power for the scaling of the loss (2 for MSE, 1 for MAE, etc)
        * distmat_mask: (N, N) mask (boolean or weights for each ij pos). optional.
    """
    # **2 ensures always positive. Later scale back to desired power
    loss = ( torch.cdist(X, X, p=p) - torch.cdist(Y, Y, p=p) )**2 
    if q != 2:
        loss = loss**(q/2)
    if distmat_mask is not None:
        loss *= distmat_mask.float()
    return loss.mean()

def rmsd_torch(X, Y):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    return torch.sqrt( torch.mean((X - Y)**2, axis=(-1, -2)) )

def rmsd_numpy(X, Y):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    return np.sqrt( np.mean((X - Y)**2, axis=(-1, -2)) )

def gdt_torch(X, Y, cutoffs, weights=None):
    """ Assumes x,y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    device = X.device
    if weights is None:
        weights = torch.ones(1,len(cutoffs))
    else:
        weights = torch.tensor([weights]).to(device)
    # set zeros and fill with values
    GDT = torch.zeros(X.shape[0], len(cutoffs), device=device)
    dist = ((X - Y)**2).sum(dim=1).sqrt()
    # iterate over thresholds
    for i,cutoff in enumerate(cutoffs):
        GDT[:, i] = (dist <= cutoff).float().mean(dim=-1)
    # weighted mean
    return (GDT*weights).mean(-1)

def gdt_numpy(X, Y, cutoffs, weights=None):
    """ Assumes x,y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    if weights is None:
        weights = np.ones( (1,len(cutoffs)) )
    else:
        weights = np.array([weights])
    # set zeros and fill with values
    GDT = np.zeros( (X.shape[0], len(cutoffs)) )
    dist = np.sqrt( ((X - Y)**2).sum(axis=1) )
    # iterate over thresholds
    for i,cutoff in enumerate(cutoffs):
        GDT[:, i] = (dist <= cutoff).mean(axis=-1)
    # weighted mean
    return (GDT*weights).mean(-1)

def tmscore_torch(X, Y):
    """ Assumes x,y are both (B x D x N). see below for wrapper. """
    L = X.shape[-1]
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    # get distance
    dist = ((X - Y)**2).sum(dim=1).sqrt()
    # formula (see wrapper for source): 
    return (1 / (1 + (dist/d0)**2)).mean(dim=-1)

def tmscore_numpy(X, Y):
    """ Assumes x,y are both (B x D x N). see below for wrapper. """
    L = X.shape[-1]
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    # get distance
    dist = np.sqrt( ((X - Y)**2).sum(axis=1) )
    # formula (see wrapper for source): 
    return (1 / (1 + (dist/d0)**2)).mean(axis=-1)


def mdscaling_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5,
                    fix_mirror=True, N_mask=None, CA_mask=None, C_mask=None, verbose=2):
    """ Handles the specifics of MDS for proteins (mirrors, ...) """
    # batched mds for full parallel 
    preds, stresses = mds_torch(pre_dist_mat, weights=weights,iters=iters, 
                                              tol=tol, verbose=verbose)
    if not fix_mirror:
        return preds, stresses

    # no need to caculate multiple mirrors - just correct Z axis
    phi_ratios = calc_phis_torch(preds, N_mask, CA_mask, C_mask, prop=True)
    to_correct = torch.nonzero( (phi_ratios < 0.5) ).view(-1)
    # fix mirrors by (-1)*Z if more (+) than (-) phi angles
    preds[to_correct, -1] = (-1)*preds[to_correct, -1]
    if verbose == 2:
        print("Corrected mirror idxs:", to_correct)
            
    return preds, stresses


def mdscaling_numpy(pre_dist_mat, weights=None, iters=10, tol=1e-5,
                    fix_mirror=True, N_mask=None, CA_mask=None, C_mask=None, verbose=2):
    """ Handles the specifics of MDS for proteins (mirrors, ...) """
    # batched mds for full parallel 
    preds, stresses = mds_numpy(pre_dist_mat, weights=weights,iters=iters, 
                                              tol=tol, verbose=verbose)
    if not fix_mirror:
        return preds, stresses

    # no need to caculate multiple mirrors - just correct Z axis
    phi_ratios = calc_phis_numpy(preds, N_mask, CA_mask, C_mask, prop=True)
    for i,pred in enumerate(preds):
        # fix mirrors by (-1)*Z if more (+) than (-) phi angles
        if phi_ratios < 0.5:
            preds[i, -1] = (-1)*preds[i, -1]
            if verbose == 2:
                print("Corrected mirror in struct no.", i)

    return preds, stresses


def lddt_ca_torch(true_coords, pred_coords, cloud_mask, r_0=15.):
    """ Computes the lddt score for each C_alpha.
        https://academic.oup.com/bioinformatics/article/29/21/2722/195896
        Inputs: 
        * true_coords: (b, l, c, d) in sidechainnet format.
        * pred_coords: (b, l, c, d) in sidechainnet format.
        * cloud_mask : (b, l, c) adapted for scn format.
        * r_0: float. maximum inclusion radius in reference struct.
        Outputs:
        * (b, l) lddt for c_alpha scores (ranging between 0 and 1)
        See wrapper below.
    """
    device, dtype = true_coords.device, true_coords.type()
    thresholds = torch.tensor([0.5, 1, 2, 4], device=device).type(dtype)
    # adapt masks
    cloud_mask = cloud_mask.bool().cpu()
    c_alpha_mask  = torch.zeros(cloud_mask.shape[1:], device=device).bool() # doesn't have batch dim
    c_alpha_mask[..., 1] = True
    # container for c_alpha scores (between 0,1)
    wrapper = torch.zeros(true_coords.shape[:2], device=device).type(dtype)

    for bi, seq in enumerate(true_coords):
        # select atoms for study
        c_alphas = cloud_mask[bi]*c_alpha_mask # only pick c_alpha positions
        selected_pred = pred_coords[bi, c_alphas, :] 
        selected_target = true_coords[bi, c_alphas, :]
        # get number under distance
        dist_mat_pred = torch.cdist(selected_pred, selected_pred, p=2)
        dist_mat_target = torch.cdist(selected_target, selected_target, p=2) 
        under_r0_target = dist_mat_target < r_0
        compare_dists = torch.abs(dist_mat_pred - dist_mat_target)[under_r0_target]
        # measure diff below threshold
        score = torch.zeros_like(under_r0_target).float()
        max_score = torch.zeros_like(under_r0_target).float()
        max_score[under_r0_target] = 4.
        # measure under how many thresholds
        score[under_r0_target] = thresholds.shape[0] - \
                                 torch.bucketize( compare_dists, boundaries=thresholds ).float()
        # dont include diagonal
        l_mask = c_alphas.float().sum(dim=-1).bool()
        wrapper[bi, l_mask] = ( score.sum(dim=-1) - thresholds.shape[0] ) / \
                              ( max_score.sum(dim=-1) - thresholds.shape[0] )

    return wrapper


################
### WRAPPERS ###
################

@set_backend_kwarg
@invoke_torch_or_numpy(mdscaling_torch, mdscaling_numpy)
def MDScaling(pre_dist_mat, **kwargs):
    """ Gets distance matrix (-ces). Outputs 3d.  
        Assumes (for now) distrogram is (N x N) and symmetric.
        For support of ditograms: see `center_distogram_torch()`
        Inputs:
        * pre_dist_mat: (1, N, N) distance matrix.
        * weights: optional. (N x N) pairwise relative weights .
        * iters: number of iterations to run the algorithm on
        * tol: relative tolerance at which to stop the algorithm if no better
               improvement is achieved
        * backend: one of ["numpy", "torch", "auto"] for backend choice
        * fix_mirror: int. number of iterations to run the 3d generation and
                      pick the best mirror (highest number of negative phis)
        * N_mask: indexing array/tensor for indices of backbone N.
                  Only used if fix_mirror > 0.
        * CA_mask: indexing array/tensor for indices of backbone C_alpha.
                   Only used if fix_mirror > 0.
        * verbose: whether to print logs
        Outputs:
        * best_3d_coords: (3 x N)
        * historic_stress: (timesteps, )
    """
    pre_dist_mat = expand_dims_to(pre_dist_mat, 3 - len(pre_dist_mat.shape))
    return pre_dist_mat, kwargs

@expand_arg_dims(dim_len = 2)
@set_backend_kwarg
@invoke_torch_or_numpy(kabsch_torch, kabsch_numpy)
def Kabsch(A, B):
    """ Returns Kabsch-rotated matrices resulting
        from aligning A into B.
        Adapted from: https://github.com/charnley/rmsd/
        * Inputs: 
            * A,B are (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of shape (3 x N)
    """
    # run calcs - pick the 0th bc an additional dim was created
    return A, B

@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(rmsd_torch, rmsd_numpy)
def RMSD(A, B):
    """ Returns RMSD score as defined here (lower is better):
        https://en.wikipedia.org/wiki/
        Root-mean-square_deviation_of_atomic_positions
        * Inputs: 
            * A,B are (B x 3 x N) or (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of size (B,)
    """
    return A, B

@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(gdt_torch, gdt_numpy)
def GDT(A, B, *, mode="TS", cutoffs=[1,2,4,8], weights=None):
    """ Returns GDT score as defined here (highre is better):
        Supports both TS and HA
        http://predictioncenter.org/casp12/doc/help.html
        * Inputs:
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * cutoffs: defines thresholds for gdt
            * weights: list containing the weights
            * mode: one of ["numpy", "torch", "auto"] for backend
        * Outputs: tensor/array of size (B,)
    """
    # define cutoffs for each type of gdt and weights
    cutoffs = [0.5,1,2,4] if mode in ["HA", "ha"] else [1,2,4,8]
    # calculate GDT
    return A, B, cutoffs, {'weights': weights}

@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(tmscore_torch, tmscore_numpy)
def TMscore(A, B):
    """ Returns TMscore as defined here (higher is better):
        >0.5 (likely) >0.6 (highly likely) same folding. 
        = 0.2. https://en.wikipedia.org/wiki/Template_modeling_score
        Warning! It's not exactly the code in:
        https://zhanglab.ccmb.med.umich.edu/TM-score/TMscore.cpp
        but will suffice for now. 
        Inputs: 
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * mode: one of ["numpy", "torch", "auto"] for backend
        Outputs: tensor/array of size (B,)
    """
    return A, B
