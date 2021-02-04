import torch
from torch import nn
from alphafold2_pytorch.alphafold2 import Attention, FeedForward
from alphafold2_pytorch.reversible import ReversibleSequence, ReversibleSelfAttnBlock, ReversibleCrossAttnBlock

def test_reversible():
    dim = 32
    num_layers = 4

    seq = torch.randn(1, 16 * 16, dim)
    msa = torch.randn(1, 16, dim)

    mask = torch.ones(1, 16 * 16).bool()
    msa_mask = torch.ones(1, 16).bool()

    # reversible modules

    layers = nn.ModuleList([])
    for _ in range(4):
        layers.append(nn.ModuleList([
           Attention(dim),
           FeedForward(dim),
           Attention(dim),
           FeedForward(dim) 
        ]))

        layers.append(nn.ModuleList([
           Attention(dim),
           FeedForward(dim),
           Attention(dim),
           FeedForward(dim) 
        ]))

    layers = ReversibleSequence(layers)

    seq1 = seq.clone().requires_grad_()
    msa1 = msa.clone().requires_grad_()
    
    seq2 = seq.clone().requires_grad_()
    msa2 = msa.clone().requires_grad_()

    def loss_fn(*args):
        return sum(map(lambda t: t.sum(dim = 1), args)).sum()

    loss_fn(*layers(seq1, msa1, mask = mask, msa_mask = msa_mask, reverse = True)).backward()
    loss_fn(*layers(seq2, msa2, mask = mask, msa_mask = msa_mask, reverse = False)).backward()

    assert torch.allclose(seq2.grad, seq1.grad, atol=1e-3)
    assert torch.allclose(msa1.grad, msa2.grad, atol=1e-3)
