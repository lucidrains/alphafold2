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
