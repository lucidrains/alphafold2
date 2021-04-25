import torch
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

def test_anglegrams():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_angles = True
    )

    seq = torch.randint(0, 21, (2, 128))
    msa = torch.randint(0, 21, (2, 5, 64))
    mask = torch.ones_like(seq).bool()
    msa_mask = torch.ones_like(msa).bool()

    distogram, theta, phi, omega = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )
    assert True

def test_msa_tie_row_attn():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
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
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        attn_types = ('full', 'intra_attn', 'seq_only')
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

def test_coords_se3():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        num_backbone_atoms = 3,
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
        structure_module_knn = 2
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

    assert coords.shape == (2, 16 * 14, 3), 'must output coordinates'

def test_edges_to_equivariant_network():
    model = Alphafold2(
        dim = 32,
        depth = 1,
        heads = 2,
        dim_head = 32,
        use_se3_transformer = False,
        predict_coords = True,
        predict_angles = True,
        num_backbone_atoms = 3
    )

    seq = torch.randint(0, 21, (2, 16))
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

def test_real_value_distance_with_coords():
    model = Alphafold2(
        dim = 32,
        depth = 1,
        heads = 2,
        dim_head = 16,
        predict_coords = True,
        predict_real_value_distances = True,
        num_backbone_atoms = 3,
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
        structure_module_knn = 2
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

    assert coords.shape == (2, 16 * 14, 3), 'must output coordinates'

def test_coords_se3_backwards():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        num_backbone_atoms = 3,
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
        structure_module_knn = 2
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

    coords.sum().backward()
    assert True, 'must be able to go backwards through MDS and center distogram'

def test_coords_En():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 2,
        dim_head = 32,
        use_se3_transformer = False,
        predict_coords = True,
        num_backbone_atoms = 3
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
    # get masks : cloud is all points in prot. chain is all for which we have labels
    cloud_mask = scn_cloud_mask(seq, boolean = True)
    flat_cloud_mask = rearrange(cloud_mask, 'b l c -> b (l c)')
    chain_mask = (mask.unsqueeze(-1) * cloud_mask)
    flat_chain_mask = rearrange(chain_mask, 'b l c -> b (l c)')

    # put in sidechainnet format
    wrapper = torch.zeros(*cloud_mask.shape, 3).to(coords.device).type(coords.type())
    wrapper[cloud_mask] = coords[flat_cloud_mask]

    assert wrapper[chain_mask].shape == coords[flat_chain_mask].shape, 'must output coordinates'


def test_coords_En_backwards():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 2,
        dim_head = 32,
        use_se3_transformer = False,
        predict_coords = True,
        num_backbone_atoms = 3
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

    coords.sum().backward()
    assert True, 'must be able to go backwards through MDS and center distogram'


def test_confidence_En():
    model = Alphafold2(
        dim = 256,
        depth = 1,
        heads = 2,
        dim_head = 32,
        use_se3_transformer = False,
        predict_coords = True,
        num_backbone_atoms = 3
    )

    seq = torch.randint(0, 21, (2, 16))
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
    
    assert coords.shape[:-1] == confidences.shape[:-1]


def test_reversible():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
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