import torch
from alphafold2_pytorch.alphafold2 import Alphafold2

def test_main():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 8,
        dim_head = 64
    )

    seq = torch.randint(0, 21, (2, 128))
    msa = torch.randint(0, 21, (2, 5, 64))
    mask = torch.ones_like(seq).bool()
    msa_mask = torch.ones_like(msa).bool()

    distogram = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )
    assert True

def test_msa_tie_row_attn():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 8,
        dim_head = 64,
        msa_tie_row_attn = True
    )

    seq = torch.randint(0, 21, (2, 128))
    msa = torch.randint(0, 21, (2, 5, 64))
    mask = torch.ones_like(seq).bool()
    msa_mask = torch.ones_like(msa).bool()

    distogram = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )
    assert True

def test_templates():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 8,
        dim_head = 64
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 32))
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

def test_coords():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 8,
        dim_head = 64,
        predict_coords = True
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 32))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 16, 3), 'must output coordinates'

def test_coords_backwards():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 8,
        dim_head = 64,
        predict_coords = True
    )

    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 32))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    coords.sum().backwards()
    assert True, 'must be able to go backwards through MDS and center distogram'

def test_reversible():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 8,
        dim_head = 64,
        reversible = True
    )

    seq = torch.randint(0, 21, (2, 128))
    msa = torch.randint(0, 21, (2, 5, 64))
    mask = torch.ones_like(seq).bool()
    msa_mask = torch.ones_like(msa).bool()

    distogram = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    distogram.sum().backward()
    assert True
