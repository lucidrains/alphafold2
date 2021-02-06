import torch
from alphafold2_pytorch.utils import *

def test_center_distogram_median():
    distogram = torch.randn(1, 128, 128, 37)
    distances, weights = center_distogram_torch(distogram, center = 'median')
    assert True

def test_mds():
    distogram = torch.randn(1, 128, 128, 37)

    distances, weights = center_distogram_torch(distogram)

    coords_3d, _ = MDScaling(distances, 
        weights = weights,
        iters = 200, 
        fix_mirror = 0
    )

    assert coords_3d.shape == (3, 128), 'coordinates must be of the right shape after MDS'

def test_tmscore():
    a = torch.randn(2, 3, 8)
    b = torch.randn(2, 3, 8)
    out = TMscore(a, b)
    assert True

def test_gdt():
    a = torch.randn(1, 3, 8)
    b = torch.randn(1, 3, 8)
    GDT(a, b, weights = 1)
    assert True
