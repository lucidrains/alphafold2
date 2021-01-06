# utils for working with 3d-protein structures
import numpy as np 
import torch

# common utils

def custom2pdb(coords, atom_names, aa_belongs):
    """ Takes a custom representation and turns into a .pdb file. """
    raise NotImplementedError("Not implemented yet. Yet to see what's our output format.")

def shape_and_backend(x,y,backend):
    """ pack here for reuse. 
        turns input into (B x D x N) and chooses backend
    """
    # auto type infer mode
    if backend == "auto":
        backend = "torch" if isinstance(x, torch.Tensor) else "numpy"
    # check shapes
    if len(x.shape) == len(y.shape):
        while len(x.shape) < 3:
            if backend == "torch":
                x = x.unsqueeze(dim=0)
                y = y.unsqueeze(dim=0)
            else:
                x = np.expand_dims(x, axis=0)
                y = np.expand_dims(y, axis=0)
    else:
        raise ValueError("Shapes of A and B must match.")

    return x,y,backend


# distogram to 3d coords: https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/manifold/_mds.py#L279

def mds_torch(distogram, probs=None, iters=10, tol=1e-5, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper. 
        Assumes (for now) distrogram is (N x N) and symmetric
    """
    N = distogram.shape[-1]
    his = []
    # init random coords
    best_stress = float("Inf") 
    best_3d_coords = 2*torch.rand(3, N) - 1
    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        dist_mat = torch.cdist(best_3d_coords.t(), best_3d_coords.t(), p=2)
        stress   = ((dist_mat - distogram)**2).sum() / 2
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[dist_mat == 0] = 1e-5
        ratio = distogram / dist_mat
        B = ratio * (-1)
        B[np.arange(N), np.arange(N)] += ratio.sum(dim=1)
        # update - double transpose. TODO: consider fix
        coords = (1. / N * torch.matmul(best_3d_coords, B))
        dis = torch.sqrt((coords ** 2).sum(axis=1)).sum()
        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if(best_stress - stress / dis) > tol:
            best_3d_coords = coords
            best_stress = (stress / dis).item()
            his.append(best_stress)
        else:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress))
            break

    return best_3d_coords, torch.tensor(his)

def mds_numpy(distogram, probs=None, iters=10, tol=1e-5, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper. 
        Assumes (for now) distrogram is (N x N) and symmetric
        Out:
        * best_3d_coords: (3 x N)
        * historic_stress 
    """
    N = distogram.shape[-1]
    his = []
    # init random coords
    best_stress = np.inf 
    best_3d_coords = 2*np.random.rand(3, N) - 1
    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        dist_mat = np.linalg.norm(np.expand_dims(best_3d_coords,1) - np.expand_dims(best_3d_coords,2), axis=0)
        stress   = ((dist_mat - distogram)**2).sum() / 2
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[dist_mat == 0] = 1e-5
        ratio = distogram / dist_mat
        B = ratio * (-1)
        B[np.arange(N), np.arange(N)] += ratio.sum(axis=1)
        # update - double transpose. TODO: consider fix
        coords = (1. / N * np.dot(best_3d_coords, B))
        dis = np.sqrt((coords ** 2).sum(axis=1)).sum()
        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if(best_stress - stress / dis) > tol:
            best_3d_coords = coords
            best_stress = stress / dis
            his.append(best_stress)
        else:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress))
            break

    return best_3d_coords, np.array(his)

# TODO: test
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

def fix_mirrors_torch(preds, N_mask, CA_mask, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Ca{-1}, N, Ca{0}) and (Ca{0}, N{+1}, C_a{+1})
    """ 
    preds_ = torch.cat([x[0].detach().unsqueeze(0) for x in preds], dim=0)
    ns = torch.transpose(preds_, -1, -2)[:, N_mask][:, 1:]
    cs = torch.transpose(preds_, -1, -2)[:, CA_mask]
    # compute phis and count lower than 0s
    phis_count = []
    for i in range(cs.shape[0]):
        phis = []
        for j in range(1, cs.shape[1]-1):
            phis.append( get_dihedral_torch(cs[i,j-1], ns[i,j], cs[i,j], ns[i,j+1], cs[i,j+1]) )
        phis_count.append( (torch.tensor(phis)<0).float().sum() )
    # debugging/testing if arg passed
    if verbose:
        print("Negative phis:", phis_count)
    return preds[torch.argmax(torch.tensor(phis_count))]

def fix_mirrors_numpy(preds, N_mask, CA_mask, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Ca{-1}, N, Ca{0}) and (Ca{0}, N{+1}, C_a{+1})
    """ 
    preds_ = np.array([x[0] for x in preds])
    ns = np.transpose(preds_, (0, 2, 1))[N_mask][1:]
    cs =  np.transpose(preds_, (0, 2, 1))[CA_mask]
    # compute phis and count lower than 0s
    phis_count = []
    for i in range(cs.shape[0]):
        phis = []
        for j in range(1, cs.shape[1]-1):
            phis.append( get_dihedral_numpy(cs[i,j-1], ns[i,j], cs[i,j], ns[i,j+1], cs[i,j+1]) )
        phis_count.append( (np.array(phis)<0).sum() )
    # debugging/testing if arg passed
    if verbose:
        print("Negative phis:", phis_count)
    return preds[np.argmax(np.array(phis_count))]


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
    V, S, W = torch.svd(C)
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


################
### WRAPPERS ###
################

def MDScaling(distogram, iters=10, tol=1e-5, backend="auto",
              fix_mirror=0, N_mask=None, CA_mask=None, verbose=2):
    """ Gets distance matrix (-ces). Outputs 3d.  
        Assumes (for now) distrogram is (N x N) and symmetric.
        Inputs:
        * distogram: (N x N) distance matrix. TODO: support probabilities
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
        * historic_stress: list
    """
    if backend == "auto":
        if isinstance(distogram, torch.Tensor):
            backend = "torch"
        else:
            backend = "numpy"
    # run calcs     
    if backend == "torch":
        preds = [mds_torch(distogram, iters=iters, tol=tol, verbose=verbose) \
                 for i in range( max(1,fix_mirror) )]
        if not fix_mirror:
            return preds[0]
        else:
            return fix_mirrors_torch(preds, N_mask, CA_mask)
    else:
        preds = [mds_numpy(distogram, iters=iters, tol=tol, verbose=verbose) \
                 for i in range(max(1,fix_mirror))]
        if not fix_mirror:
            return pred[0]
        else:
            return fix_mirrors_numpy(preds, N_mask, CA_mask)


def Kabsch(A, B, backend="auto"):
    """ Returns Kabsch-rotated matrices resulting
        from aligning A into B.
        Adapted from: https://github.com/charnley/rmsd/
        * Inputs: 
            * A,B are (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of shape (3 x N)
    """
    A, B, backend = shape_and_backend(A, B, backend)
    # run calcs - pick the 0th bc an additional dim was created
    if backend == "torch":
        return kabsch_torch(A[0], B[0])
    else:
        return kabsch_numpy(A[0], B[0])


def RMSD(A, B, backend="auto"):
    """ Returns RMSD score as defined here (lower is better):
        https://en.wikipedia.org/wiki/
        Root-mean-square_deviation_of_atomic_positions
        * Inputs: 
            * A,B are (B x 3 x N) or (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of size (B,)
    """
    A, B, backend = shape_and_backend(A, B, backend)
    # run calcs
    if backend == "torch":
        return rmsd_torch(A, B)
    else:
        return rmsd_numpy(A, B)


def GDT(A,B, mode="TS", cutoffs=[1,2,4,8], weights=None, backend="auto"):
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
    A, B, backend = shape_and_backend(A, B, backend)
    # define cutoffs for each type of gdt and weights
    cutoffs = [0.5,1,2,4] if mode in ["HA", "ha"] else [1,2,4,8]
    # calculate GDT
    if backend == "torch":
        return gdt_torch(A, B, cutoffs, weights=weights)
    else:
        return gdt_numpy(A, B, cutoffs, weights=weights)


def TMscore(A,B,backend="auto"):
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
    A, B, backend = shape_and_backend(A, B, backend)
    # run calcs
    if backend == "torch":
        return tmscore_torch(A, B)
    else:
        return tmscore_numpy(A, B)


