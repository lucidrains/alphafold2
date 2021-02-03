import torch
from alphafold2_pytorch.utils import *

def test_mds():
    distogram = torch.randn(1, 128, 128, 37)

    distances, weights = center_distogram_torch(distogram)

    coords_3d, _ = MDScaling(distances, 
        weights = weights,
        iters = 200, 
        fix_mirror = 0
    )

    assert coords_3d.shape == (3, 128), 'coordinates must be of the right shape after MDS'
