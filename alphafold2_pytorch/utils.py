# utils for working with 3d-protein structures
import os
import numpy as np 
import torch
from functools import wraps
from einops import rearrange, repeat

# bio
import mdtraj
try:
    from sidechainnet.sequence.utils import VOCAB
    from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
    from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, SC_BUILD_INFO
    from sidechainnet.structure.StructureBuilder import _get_residue_build_iter
except:
    NUM_COORDS_PER_RES = 14
    GLOBAL_PAD_CHAR = 0

# own
from alphafold2_pytorch.alphafold2 import DISTOGRAM_BUCKETS

# constants: same as in alphafold2.py

DISTANCE_THRESHOLDS = torch.linspace(2, 20, steps = DISTOGRAM_BUCKETS)

# decorators

def set_backend_kwarg(fn):
    @wraps(fn)
    def inner(*args, backend = 'auto', **kwargs):
        if backend == 'auto':
            backend = 'torch' if isinstance(args[0], torch.Tensor) else 'numpy'
        kwargs.update(backend = backend)
        print('settinb backend kwargs', backend, kwargs)
        return fn(*args, **kwargs)
    return inner

def cast_num_dimensions(dim_len = 3):
    """ pack here for reuse. 
        turns input into (B x D x N)
    """
    def outer(fn):
        @wraps(fn)
        def inner(x, y, **kwargs):
            assert len(x.shape) == len(y.shape), "Shapes of A and B must match."
            remaining_len = len(x.shape) - dim_len
            x = x.reshape(*((1,) * remaining_len), *x.shape) # will work with both torch and numpy
            y = y.reshape(*((1,) * remaining_len), *y.shape)
            return fn(x, y, **kwargs)
        return inner
    return outer

def invoke_torch_or_numpy(torch_fn, numpy_fn):
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            print('getting')
            backend = kwargs.pop('backend')
            passed_args, passed_kwargs = fn(*args, **kwargs)
            backend_fn = torch_fn if backend == 'torch' else numpy_fn
            return backend_fn(*passed_args, **passed_kwargs)
        return inner
    return outer

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
    os.system("curl https://files.rcsb.org/download/{0}.pdb > {1}".format(name, route))
    return route

def clean_pdb(name, route=None, chain_num=None):
    """ Cleans the structure to only leave the important part.
        Inputs: 
        * name: str. route of the input .pdb file
        * route: str. route of the output. will overwrite input if not provided
        * chain_num: int. index of chain to select (1-indexed as pdb files)
        Output: route of destin file.
    """
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
        chain_idxs = raw_prot.topology.select("chainid == {0}".format(chain.index))
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
    # convert to numpy
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    # ensure (1, N, 3)
    if coords.shape[1] == 3:
        coords = coords.T
    coords = np.newaxis(coords, axis=0)
    # get pdb id and chain num
    pdb_name, china_num = proteinnet_id.split("#")[-1].split("_")[:-1]
    pdb_destin = "/".join(route.split("/")[:-1])+"/"+pdb_name+".pdb"
    # download pdb file and select appropiate 
    download_pdb(pdb_name, pdb_destin)
    clean_pdb(pdb_destin, chain_num=chain_num)
    # load trajectory scaffold and replace coordinates - assumes same order
    scaffold = mdtraj.load_pdb(pdb_destin)
    scaffold.xyz = coords
    scaffold.save(route)
    return pdb_destin, route


# sidechainnet utils

def scn_cloud_mask(scn_seq, bool=True):
    """ Gets the boolean mask atom positions (not all aas have same atoms). 
        Inputs: 
        * scn_seq: (batch, length) sequence as provided by Sidechainnet package
        * bool: whether to return as array of idxs or boolean values
        * mask: (batch, length). current mask to build a custom chain_mask 
        Outputs: (batch, length, NUM_COORDS_PER_RES) boolean mask 
    """
    # scaffolds 
    mask = torch.zeros(*scn_seq.shape, NUM_COORDS_PER_RES, device=snc_seq.device)
    chain_mask = []
    # fill 
    for n in range(len(masks)):
        for i,aa in enumerate(scn_seq.cpu().numpy()):
            # get num of atom positions - backbone is 4: ...N-C-C(=O)...
            n_atoms = 4+len( SC_BUILD_INFO[VOCAB.int2chars(x)]["atom-names"] )
            mask[n, i, :n_atoms] = 1
    return mask.bool()

def scn_backbone_mask(scn_seq, bool=True, l_aa=NUM_COORDS_PER_RES):
    """ Gets the boolean mask for N and CA positions. 
        Inputs: 
        * scn_seq: sequence as provided by Sidechainnet package
        * bool: whether to return as array of idxs or boolean values
        Outputs: (N_mask, CA_mask)
    """
    lengths = np.arange(scn_seq.shape[-1]*l_aa)
    # N is the first atom in every AA. CA is the 2nd.
    N_mask  = lengths%l_aa == 0
    CA_mask = lengths%l_aa == 1
    if boolean:
        return N_mask, CA_mask

    return lengths[N_mask], lengths[CA_mask]

def sidechain_3d(seq, backbone, n_atoms=NUM_COORDS_PER_RES, 
                 padding=GLOBAL_PAD_CHAR, force=False):
    """ Gets a backbone of the protein, returns the whole coordinates
        with sidechains (same format as sidechainnet). Keeps differentiability.
        Inputs: 
        * seq: (L,) tensor of ints. sequence tokens.
        * backbone: (batch, L*3, 3): assume batch=1 (could be extended later).
                    Coords for (N-term, C-alpha, C-term) of every aa.
        * n_atoms: int. n of atom positions / atom. same as in sidechainnet: 14
        * padding: int. padding token. same as in sidechainnet: 0
        Outputs: whole coordinates of shape (batch, L, n_atoms, 3)
    """
    batch, length = list(backbone.shape[:2])
    new_coords = torch.ones(batch, length//3, NUM_COORDS_PER_RES, 3) * padding
    new_coords[:, :, :3] = rearrange(backbone, 'b (l back) d -> b l back d', back=3)
    # build sidechain for every aa
    for i,token in enumerate(seq):
        # position C-beta
        # 
        # # get location of (=O) spanding from previous C-term
        #
        # # get tetrahedral conformation of C-alpha, find closest to (=O), exploit D-aa
        #
        # iterate over aa atoms and place them accordingly
        for j, (bond_len, angle, torsion, atom_names) in enumerate(
            _get_residue_build_iter(token, SC_BUILD_INFO)):
            pass 
            # extend from previous atom
            #
    if force:
        return new_coords

    raise NotImplementedError("Function not implemented yet")


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
    n_bins = bins - 0.5 * (bins[2] - bins[2])
    n_bins[0]  = 1.5
    # TODO: adapt so that mean option considers IGNORE_INDEX
    n_bins[-1] = n_bins[-1]
    n_bins = n_bins.to(device)
    # calculate measures of centrality and dispersion - 
    if center == "median":
        cum_dist = torch.cumsum(distogram, dim=-1)
        medium   = 0.5 * torch.ones(*cum_dist.shape[:-1], device=device).unsqueeze(dim=-1)
        central  = torch.searchsorted(cum_dist, medium).squeeze()
        central  = n_bins[ torch.minimum(central, torch.tensor(DISTOGRAM_BUCKETS-1)).long() ]
    elif center == "mean":
        central  = (distogram * n_bins).sum(dim=-1)
    # create mask for last class - (IGNORE_INDEX)   
    mask = (central <= bins[-2].item()).float()
    # mask diagonal to 0 dist 
    diag = np.arange(shape[-2])
    central[:, diag, diag] = 0.
    # provide weights
    if wide == "var":
        dispersion = (distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1)
    elif wide == "std":
        dispersion = (distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1).sqrt()
    else:
        dispersion = torch.zeros_like(central, device=central.device)
    # rescale to 0-1. lower std / var  --> weight=1
    weights = mask / (1+dispersion)

    return central, weights

# distance matrix to 3d coords: https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/manifold/_mds.py#L279

def mds_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper. 
        Assumes (for now) distrogram is (N x N) and symmetric
        Outs: 
        * best_3d_coords: (3 x N)
        * historic_stress 
    """
    device = pre_dist_mat.device

    if weights is None:
        weights = torch.ones_like(pre_dist_mat)
    # batched MDS
    if len(pre_dist_mat.shape) < 3:
        pre_dist_mat.unsqueeze_(0)
    # start
    batch, N, _ = pre_dist_mat.shape
    his = []
    # init random coords
    best_stress = float("Inf") * torch.ones(batch, device = device)
    best_3d_coords = 2*torch.rand(batch, N, 3, device = device) - 1
    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        dist_mat = torch.cdist(best_3d_coords, best_3d_coords, p=2)
        stress   = ( weights * (dist_mat - pre_dist_mat)**2 ).sum(dim=(-1,-2)) / 2
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[dist_mat == 0] = 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = ratio * (-1)
        B[:, np.arange(N), np.arange(N)] += ratio.sum(dim=-1)
        # update
        coords = (1. / N * torch.matmul(B, best_3d_coords))
        dis = torch.norm(coords, dim=(-1, -2))
        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (best_stress - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress))
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
    N = pre_dist_mat.shape[-1]
    his = []
    # init random coords
    best_stress = np.inf 
    best_3d_coords = 2*np.random.rand(3, N) - 1
    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        dist_mat = np.linalg.norm(np.expand_dims(best_3d_coords,1) - np.expand_dims(best_3d_coords,2), axis=0)
        stress   = (( weights * (dist_mat - pre_dist_mat) )**2).sum() / 2
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[dist_mat == 0] = 1e-7
        ratio = pre_dist_mat / dist_mat
        B = ratio * (-1)
        B[np.arange(N), np.arange(N)] += ratio.sum(axis=1)
        # update - double transpose. TODO: consider fix
        coords = (1. / N * np.dot(best_3d_coords, B))
        dis = np.linalg.norm(coords)
        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (best_stress - stress / dis) <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress))
            break

        best_3d_coords = coords
        best_stress = stress / dis
        his.append(best_stress)

    return best_3d_coords, np.array(his)

def get_dihedral_torch(c1, c2, c3, c4, c5):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
    """

    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3
    u4 = c5 - c4

    return torch.atan2( torch.dot( torch.norm(u2) * u1, torch.cross(u3,u4) ),  
                        torch.dot( torch.cross(u1,u2), torch.cross(u3, u4) ) ) 

def get_dihedral_numpy(c1, c2, c3, c4, c5):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
    """

    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3
    u4 = c5 - c4

    return np.arctan2( np.dot( np.linalg.norm(u2) * u1, np.cross(u3,u4) ),  
                       np.dot( np.cross(u1,u2), np.cross(u3, u4) ) ) 

def fix_mirrors_torch(preds, stresses, N_mask, CA_mask, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Ca{-1}, N, Ca{0}) and (Ca{0}, N{+1}, C_a{+1})
    """ 
    preds_ = preds.detach()
    ns = torch.transpose(preds_, -1, -2)[:, N_mask][:, 1:]
    cs = torch.transpose(preds_, -1, -2)[:, CA_mask]
    # compute phis and count lower than 0s
    phis_count = []
    for i in range(cs.shape[0]):
        # calculate phi angles
        phis = [ get_dihedral_torch(cs[i,j-1], ns[i,j], cs[i,j], ns[i,j+1], cs[i,j+1]) \
                 for j in range(1, cs.shape[1]-1) ]

        phis_count.append( (np.array(phis)<0).sum() )

    idx = np.argmax(phis_count)
    # debugging/testing if arg passed
    if verbose:
        print("Negative phis:", phis_count, "selected", idx)
    return preds[idx].unsqueeze(0), stresses[idx]

def fix_mirrors_numpy(preds, stresses, N_mask, CA_mask, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Ca{-1}, N, Ca{0}) and (Ca{0}, N{+1}, C_a{+1})
    """ 
    ns = np.transpose(preds, (0, 2, 1))[:, N_mask][:, 1:]
    cs =  np.transpose(preds, (0, 2, 1))[:, CA_mask]
    # compute phis and count lower than 0s
    phis_count = []
    for i in range(cs.shape[0]):
        # calculate phi angles
        phis = [ get_dihedral_numpy(cs[i,j-1], ns[i,j], cs[i,j], ns[i,j+1], cs[i,j+1]) \
                 for j in range(1, cs.shape[1]-1) ]

        phis_count.append( (np.array(phis)<0).sum() )

    idx = np.argmax(phis_count)
    # debugging/testing if arg passed
    if verbose:
        print("Negative phis:", phis_count)
    return preds[idx], stresses[idx]


# alignment by centering + rotation to compute optimal RMSD
# adapted from : https://github.com/charnley/rmsd/

def kabsch_torch(X, Y):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t())
    # Optimal rotation matrix via SVD - warning! W must be transposed
    V, S, W = torch.svd(C.detach())
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W.t())
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
    if weights is None:
        weights = torch.ones(1,len(cutoffs))
    else:
        weights = torch.tensor([weights]).to(x.device)
    # set zeros and fill with values
    GDT = torch.zeros(X.shape[0], len(cutoffs), device=X.device) 
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
              fix_mirror=0, N_mask=None, CA_mask=None, verbose=2):
    # repeat for mirrors calculations
    pre_dist_mat = repeat(pre_dist_mat, 'ni nj -> m ni nj', m = max(1,fix_mirror))
    # batched mds for full parallel 
    preds, stresses = mds_torch(pre_dist_mat, weights=weights,iters=iters, 
                                              tol=tol, verbose=verbose)
    if not fix_mirror:
        return preds[0], stresses[0]

    return fix_mirrors_torch(preds, stresses, N_mask, CA_mask)

def mdscaling_numpy(pre_dist_mat, weights=None, iters=10, tol=1e-5,
              fix_mirror=0, N_mask=None, CA_mask=None, verbose=2):
    preds = [mds_numpy(pre_dist_mat, weights=weights,iters=iters, 
                                         tol=tol, verbose=verbose) \
                 for i in range( max(1,fix_mirror) )]

    if not fix_mirror:
        return preds[0]

    return fix_mirrors_numpy([x[0] for x in preds],
                             [x[1] for x in preds], N_mask, CA_mask)
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
        * distogram: (1, N, N) distance matrix.
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
    return pre_dist_mat, kwargs

@cast_num_dimensions(dim_len = 2)
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

@cast_num_dimensions()
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

@cast_num_dimensions()
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
    return (A, B, cutoffs), {'weights': weights}

@cast_num_dimensions()
@set_backend_kwarg
@invoke_torch_or_numpy(tmscore_torch, tmscore_numpy)
def TMscore(A, B):
    return A, B
