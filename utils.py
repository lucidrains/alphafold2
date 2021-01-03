# utils for working with 3d-protein structures
import numpy as np 
import torch


def shape_and_backend(x,y,backend):
    """ pack here for reuse. 
        turns input into (B x D x N) and chooses backend
    """
    # auto type infer mode
    if backend == "auto":
        backend = "torch" if isinstance(A, torch.tensor) else "numpy"
    # check shapes
    if len(x.shape) == len(y.shape):
        while len(A.shape) < 3:
            if backend == "torch":
                x = x.unsqueeze(dim=0)
                y = y.unsqueeze(dim=0)
            else:
                x = np.expand_dims(x, axis=0)
                y = np.expand_dims(y, axis=0)
    else:
        raise ValueError("Shapes of A and B must match.")

    return x,y,backend


# metrics - more formulas here: http://predictioncenter.org/casp12/doc/help.html

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
        return torch.sqrt( torch.mean((A - B)**2, axis=(-1, -2)) )
    else:
        return np.sqrt( np.mean((A - B)**2, axis=(-1, -2)) )


def GDT(A,B, mode="TS", cutoffs=[1,2,4,8], weights=None, mode="auto"):
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
    cutoffs = [0.5,1,2,4] if func == "HA" else [1,2,4,8]
    if weights is None:
        if mode == "torch":
            weights = torch.ones(1,len(weights))
        else:
            weights = np.ones((1,len(weights)))
    else: 
        if mode == "torch":
            weights = torch.tensor([weights]).to(A.device)
        else:
            weights = np.array([weights])
    # calculate GDT
    if mode == "torch":
        GDT  = torch.zeros(A.shape[0], len(cutoffs), device=A.device) 
        dist = ((A - B)**2).sum(dim=1).sqrt()
        # iterate over thresholds
        for i,cutoff in enumerate(cutoffs):
            GDT[:, i] = (dist <= cutoff).float().sum(dim=-1)
    else:
        GDT = np.zeros((A.shape[0], len(cutoffs))) 
        dist = np.sqrt((A - B)**2).sum(dim=1)
        # iterate over thresholds
        for i,cutoff in enumerate(cutoffs):
            GDT[:, i] = (dist <= cutoff).sum(dim=-1)

    # assign weights and take meanb.d
    return (GDT*weights).mean(-1)


def TMscore(A,B,backend="auto"):
    """ Returns TMscore as defined here (higher is better):
        >0.5 (likely) >0.6 (highly likely) same folding. 
        = 0.2
        https://en.wikipedia.org/wiki/Template_modeling_score
        Inputs: 
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * mode: one of ["numpy", "torch", "auto"] for backend
        Outputs: tensor/array of size (B,)
    """
    A, B, backend = shape_and_backend(A, B, backend)
    # params and distances
    L = A.shape[-1]
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    if backend == "torch":
        dist = ((A - B)**2).sum(dim=1).sqrt()
    else:
        dist = np.sqrt((A - B)**2).sum(dim=1)
    # return calc by formula
    return (1 / (1 + (dist/d0)**2)).mean(-1)


