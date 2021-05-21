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

def test_kron_cross_attn():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        cross_attn_kron_primary = True,
        cross_attn_kron_msa = True
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

def test_templates_se3():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        template_embedder_type = 'se3',
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
    assert True

def test_templates_en():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        template_embedder_type = 'en',
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

def test_coords_se3():
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
        structure_module_knn = 2
    )

    seq = torch.randint(0, 21, (2, 8))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 8 * 3, 3), 'must output coordinates'

def test_coords_backbone_with_cbeta():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        atoms = 'backbone-with-cbeta',
        predict_coords = True,
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
        structure_module_knn = 2
    )

    seq = torch.randint(0, 21, (2, 8))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 8 * 4, 3), 'must output coordinates'

def test_coords_all_atoms():
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        atoms = 'all',
        predict_coords = True,
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
        structure_module_knn = 2
    )

    seq = torch.randint(0, 21, (2, 8))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 8 * 14, 3), 'must output coordinates'

def test_custom_coords_module():

    class CustomCoords(nn.Module):
        def __init__(self, dim, structure_module_dim):
            super().__init__()
            self.to_coords = nn.Linear(dim, 3)

        def forward(
            self,
            *,
            distance_pred,
            trunk_embeds,
            num_atoms,
            cloud_mask,
            **kwargs
        ):
            coords = self.to_coords(trunk_embeds.sum(dim = 2))
            coords = repeat(coords, 'b n c -> b (n l) c', l = cloud_mask.shape[-1])
            return coords

    coords_module = CustomCoords(
        dim = 32,
        structure_module_dim = 4
    )

    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_dim = 4,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
        structure_module_knn = 2,
        coords_module = coords_module
    )

    seq = torch.randint(0, 21, (2, 8))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 8 * 3, 3), 'must output coordinates'

def test_coords_se3_with_global_nodes():
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
        structure_module_knn = 2,
        structure_num_global_nodes = 2
    )

    seq = torch.randint(0, 21, (2, 8))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 8 * 3, 3), 'must output coordinates'

def test_edges_to_equivariant_network():
    model = Alphafold2(
        dim = 32,
        depth = 1,
        heads = 2,
        dim_head = 32,
        structure_module_type = "en",
        predict_coords = True,
        predict_angles = True
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
        structure_module_dim = 1,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
        structure_module_knn = 2
    )

    seq = torch.randint(0, 21, (2, 8))
    mask = torch.ones_like(seq).bool()

    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    assert coords.shape == (2, 8 * 3, 3), 'must output coordinates'

def test_coords_se3_backwards():
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
        structure_module_knn = 1
    )

    seq = torch.randint(0, 21, (2, 8))
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

def test_coords_En():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 2,
        dim_head = 32,
        structure_module_type = "en",
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
    # get masks : cloud is all points in prot. chain is all for which we have labels
    cloud_mask = scn_cloud_mask(seq, boolean = True)
    flat_cloud_mask = rearrange(cloud_mask, 'b l c -> b (l c)')
    chain_mask = (mask.unsqueeze(-1) * cloud_mask)
    flat_chain_mask = rearrange(chain_mask, 'b l c -> b (l c)')

    assert True


def test_coords_En_backwards():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 2,
        dim_head = 32,
        structure_module_type = "en",
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

    coords.sum().backward()
    assert True, 'must be able to go backwards through MDS and center distogram'

def test_coords_egnn_backwards():
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 2,
        dim_head = 32,
        structure_module_type = "egnn",
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

    coords.sum().backward()
    assert True, 'must be able to go backwards through MDS and center distogram'


def test_confidence_En():
    model = Alphafold2(
        dim = 256,
        depth = 1,
        heads = 2,
        dim_head = 32,
        structure_module_type = "en",
        predict_coords = True
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