import torch
from torch import nn
from einops import repeat

from alphafold2_pytorch.alphafold2 import Alphafold2
from alphafold2_pytorch.utils import *

def test_main():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32
    )

    seq = torch.randint(0, 21, (2, 128))
    msa = torch.randint(0, 21, (2, 5, 128))
    mask = torch.ones_like(seq).bool()
    msa_mask = torch.ones_like(msa).bool()

    distogram = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )
    assert True

def test_no_msa():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32
    )

    seq = torch.randint(0, 21, (2, 128))
    mask = torch.ones_like(seq).bool()

    distogram = model(
        seq,
        mask = mask
    )
    assert True

def test_anglegrams():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_angles = True
    )

    seq = torch.randint(0, 21, (2, 128))
    msa = torch.randint(0, 21, (2, 5, 128))
    mask = torch.ones_like(seq).bool()
    msa_mask = torch.ones_like(msa).bool()

    distogram, theta, phi, omega = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )
    assert True

def test_templates_en():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    templates_seq = torch.randint(0, 21, (2, 2, 16))
    templates_coors = torch.randn(2, 2, 16, 3)
    templates_mask = torch.ones_like(templates_seq).bool()

    distogram = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask,
        templates_seq = templates_seq,
        templates_coors = templates_coors,
        templates_mask = templates_mask
    )
    assert True

def test_embeddings():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    embedds = torch.randn(2, 1, 16, 1280)

    # without mask
    distogram = model(
        seq,
        mask = mask,
        embedds = embedds,
        msa_mask = None
    )
    
    # with mask
    embedds_mask = torch.ones_like(embedds[..., -1]).bool()
    distogram = model(
        seq,
        mask = mask,
        embedds = embedds,
        msa_mask = embedds_mask
    )
    assert True

def test_coords():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 16, 3), 'must output coordinates'

def test_coords_backbone_with_cbeta():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 16, 3), 'must output coordinates'

def test_coords_all_atoms():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 16, 3), 'must output coordinates'

def test_mds():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 16, 3), 'must output coordinates'

def test_edges_to_equivariant_network():
    model = Alphafold2(
        dim = 32,
        depth = 1,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        predict_angles = True
    )

    seq = torch.randint(0, 21, (2, 32))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 32))
    msa_mask = torch.ones_like(msa).bool()

    coords, confidences = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask,
        return_confidence = True
    )
    assert True, 'should run without errors'

def test_coords_backwards():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    coords.sum().backward()
    assert True, 'must be able to go backwards through MDS and center distogram'

def test_confidence():
    model = Alphafold2(
        dim = 256,
        depth = 1,
        heads = 2,
        dim_head = 32,
        predict_coords = True
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords, confidences = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask,
        return_confidence = True
    )
    
    assert coords.shape[:-1] == confidences.shape[:-1]
