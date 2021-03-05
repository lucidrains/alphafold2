<img src="./images/alphafold2.png" width="600px"></img>

## Alphafold2 - Pytorch (wip)

To eventually become an unofficial working Pytorch implementation of <a href="https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology">Alphafold2</a>, the breathtaking attention network that solved CASP14. Will be gradually implemented as more details of the architecture is released.

Once this is replicated, I intend to fold all available amino acid sequences out there in-silico and release it as an academic torrent, to further science. If you are interested in replication efforts, please drop by #alphafold at this <a href="https://discord.com/invite/vtRgjbM">Discord channel</a>

## Install

```bash
$ pip install alphafold2-pytorch
```

## Usage

Predicting distogram, like Alphafold-1, but with attention

```python
import torch
from alphafold2_pytorch import Alphafold2
from alphafold2_pytorch.utils import MDScaling, center_distogram_torch

model = Alphafold2(
    dim = 256,
    depth = 2,
    heads = 8,
    dim_head = 64,
    reversible = False  # set this to True for fully reversible self / cross attention for the trunk
).cuda()

seq = torch.randint(0, 21, (1, 128)).cuda()
msa = torch.randint(0, 21, (1, 5, 64)).cuda()
mask = torch.ones_like(seq).bool().cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

distogram = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask
) # (1, 128, 128, 37)
```

## Predicting Coordinates (wip)

Fabian's <a href="https://arxiv.org/abs/2102.13419">recent paper</a> suggests iteratively feeding the coordinates back into SE3 Transformer, weight shared, may work. I have decided to execute based on this idea, even though it is still up in the air how it actually works.

You can also use <a href="https://github.com/lucidrains/En-transformer">E(n)-Transformer</a> for the refinement, if you are in the experimental mood (paper just came out a week ago).

```python
import torch
from alphafold2_pytorch import Alphafold2

model = Alphafold2(
    dim = 256,
    depth = 2,
    heads = 8,
    dim_head = 64,
    predict_coords = True,
    use_se3_transformer = True,             # use SE3 Transformer - if set to False, will use E(n)-Transformer, Victor and Max Welling's new paper
    num_backbone_atoms = 3,                 # C, Ca, N coordinates
    structure_module_dim = 4,               # se3 transformer dimension
    structure_module_depth = 1,             # depth
    structure_module_heads = 1,             # heads
    structure_module_dim_head = 16,         # dimension of heads
    structure_module_refinement_iters = 2   # number of equivariant coordinate refinement iterations
).cuda()

seq = torch.randint(0, 21, (2, 64)).cuda()
msa = torch.randint(0, 21, (2, 5, 32)).cuda()
mask = torch.ones_like(seq).bool().cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

coords, atom_mask = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask
) # (2, 64 * 3, 3)  <-- 3 atoms per residue
```

## Sparse Attention

You can train with Microsoft Deepspeed's <a href="https://www.deepspeed.ai/news/2020/09/08/sparse-attention.html">Sparse Attention</a>, but you will have to endure the installation process. It is two-steps.

First, you need to install Deepspeed with Sparse Attention

```bash
$ sh install_deepspeed.sh
```

Next, you need to install the pip package `triton`

```bash
$ pip install triton
```

If both of the above succeeded, now you can train with Sparse Attention!

Sadly, the sparse attention is only supported for self attention, and not cross attention. I will bring in a different solution for making cross attention performant.

```python
model = Alphafold2(
    dim = 256,
    depth = 12,
    heads = 8,
    dim_head = 64,
    max_seq_len = 2048,                   # the maximum sequence length, this is required for sparse attention. the input cannot exceed what is set here
    sparse_self_attn = (True, False) * 6  # interleave sparse and full attention for all 12 layers
).cuda()
```

## Memory Compressed Attention

To save on memory for cross attention, you can set a compression ratio for the key / values, following the scheme laid out in <a href="https://arxiv.org/abs/1801.10198">this paper</a>. A compression ratio of 2-4 is usually acceptable.

```python
model = Alphafold2(
    dim = 256,
    depth = 12,
    heads = 8,
    dim_head = 64,
    cross_attn_compress_ratio = 3
).cuda()
```

## MSA processing in Trunk

<img src="./images/msa-transformer-diagram.png" width="500px"></img>

A <a href="https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1">new paper</a> by <a href="https://github.com/rmrao">Roshan Rao</a> proposes using axial attention for pretraining on MSA's. Given the strong results, this repository will use the same scheme in the trunk, specifically for the MSA self-attention.

You can also tie the row attentions of the MSA with the `msa_tie_row_attn = True` setting on initialization of `Alphafold2`. However, in order to use this, you must make sure that none of the rows in the batch of MSA is made of padding. In other words, your `msa_mask` must be completely set to `True`

```python
model = Alphafold2(
    dim = 256,
    depth = 2,
    heads = 8,
    dim_head = 64,
    msa_tie_row_attn = True # just set this to true, but batches of MSA must not contain any row that is all padding
)
```

## Template processing in Trunk (wip)

Template processing is also largely done with axial attention, with cross attention done along the number of templates dimension. This largely follows the same scheme as in the recent all-attention approach to video classification as shown <a href="https://github.com/lucidrains/TimeSformer-pytorch">here</a>.

```python
import torch
from alphafold2_pytorch import Alphafold2

model = Alphafold2(
    dim = 256,
    depth = 5,
    heads = 8,
    dim_head = 64,
    reversible = True,
    sparse_self_attn = False,
    max_seq_len = 256,
    cross_attn_compress_ratio = 3
).cuda()

seq = torch.randint(0, 21, (1, 16)).cuda()
mask = torch.ones_like(seq).bool().cuda()

msa = torch.randint(0, 21, (1, 10, 16)).cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

templates_seq = torch.randint(0, 21, (1, 2, 16)).cuda()
templates_coors = torch.randint(0, 37, (1, 2, 16, 3)).cuda()
templates_mask = torch.ones_like(templates_seq).bool().cuda()

distogram = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask,
    templates_seq = templates_seq,
    templates_coors = templates_coors,
    templates_mask = templates_mask
)
```

If sidechain information is also present, in the form of the unit vector between the C and C-alpha coordinates of each residue, you can also pass it in as follows.

```python
import torch
from alphafold2_pytorch import Alphafold2

model = Alphafold2(
    dim = 256,
    depth = 5,
    heads = 8,
    dim_head = 64,
    reversible = True,
    sparse_self_attn = False,
    max_seq_len = 256,
    cross_attn_compress_ratio = 3
).cuda()

seq = torch.randint(0, 21, (1, 16)).cuda()
mask = torch.ones_like(seq).bool().cuda()

msa = torch.randint(0, 21, (1, 10, 16)).cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

templates_seq = torch.randint(0, 21, (1, 2, 16)).cuda()
templates_coors = torch.randn(1, 2, 16, 3).cuda()
templates_mask = torch.ones_like(templates_seq).bool().cuda()

templates_sidechains = torch.randn(1, 2, 16, 3).cuda() # unit vectors of difference of C and C-alpha coordinates

distogram = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask,
    templates_seq = templates_seq,
    templates_mask = templates_mask,
    templates_coors = templates_coors,
    templates_sidechains = templates_sidechains
)
```

####  Templates Todo

- [ ] build template self and cross attention into the main reversible network

At the moment, it goes through its own round of self-cross attention prior to the main MSA <-> sequence self cross attention, just for demonstration purposes.

- [x] allow the main network to take care of binning raw template distograms, to save users from having to do the binning logic

- [x] incorporate template sidechain information, as unit vectors of difference between C and C-alpha coordinates. use either <a href="https://github.com/lucidrains/geometric-vector-perceptron">GVP</a> or one-layer, one-headed SE3 Transformers for encoding

## Equivariant Attention

There are two equivariant self attention libraries that I have prepared for the purposes of replication. One is the implementation by Fabian Fuchs as detailed in a <a href="https://fabianfuchsml.github.io/alphafold2/">speculatory blogpost</a>. The other is from a recent paper from Deepmind, claiming their approach is better than using irreducible representations.

- <a href="https://github.com/lucidrains/se3-transformer-pytorch">SE3 Transformer</a>
- <a href="https://github.com/lucidrains/lie-transformer-pytorch">Lie Transformer</a>

A new paper from Welling uses invariant features for E(n) equivariance, reaching SOTA and outperforming SE3 Transformer at a number of benchmarks, while being much faster. You can use this by simply setting `use_se3_transformer = False` on Alphafold2 initialization.

- <a href="https://github.com/lucidrains/En-transformer">E(n)-Transformer</a>

## Testing

```bash
$ python setup.py test
```

## Data

This library will use the awesome work by <a href="http://github.com/jonathanking">Jonathan King</a> at <a href="https://github.com/jonathanking/sidechainnet">this repository</a>.

To install

```bash
$ pip install git+https://github.com/jonathanking/sidechainnet.git
```

Or

```bash
$ git clone https://github.com/jonathanking/sidechainnet.git
$ cd sidechainnet && pip install -e .
```

## Speculation

https://xukui.cn/alphafold2.html

https://moalquraishi.wordpress.com/2020/12/08/alphafold2-casp14-it-feels-like-ones-child-has-left-home/

<img src="./images/science.png"></img>

<img src="./images/reddit.png"></img>

Developments from competing labs

https://www.biorxiv.org/content/10.1101/2020.12.10.419994v1.full.pdf

<a href="./images/tFold.pdf">tFold presentation, from Tencent AI labs</a>

## External packages

* **Final step** - <a href="https://graylab.jhu.edu/PyRosetta.documentation/pyrosetta.rosetta.protocols.relax.html#pyrosetta.rosetta.protocols.relax.FastRelax">Fast Relax</a> - **Installation Instructions**:
    * Download the pyrosetta wheel from: http://www.pyrosetta.org/dow (select appropiate version) - beware the file is heavy (approx 1.2 Gb)
        * Ask for username and password to `@hypnopump` in the Discord
    * Bash > `cd downloads_folder` > `pip install pyrosetta_wheel_filename.whl`

<a href="https://parmed.github.io/ParmEd/html/omm_amber.html">OpenMM Amber</a>

## Citations

```bibtex
@misc{unpublished2021alphafold2,
    title   = {Alphafold2},
    author  = {John Jumper},
    year    = {2020},
    archivePrefix = {arXiv},
    primaryClass = {q-bio.BM}
}
```

```bibtex
@article{Rao2021.02.12.430858,
    author  = {Rao, Roshan and Liu, Jason and Verkuil, Robert and Meier, Joshua and Canny, John F. and Abbeel, Pieter and Sercu, Tom and Rives, Alexander},
    title   = {MSA Transformer},
    year    = {2021},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2021/02/13/2021.02.12.430858},
    journal = {bioRxiv}
}
```

```bibtex
@misc{king2020sidechainnet,
    title   = {SidechainNet: An All-Atom Protein Structure Dataset for Machine Learning}, 
    author  = {Jonathan E. King and David Ryan Koes},
    year    = {2020},
    eprint  = {2010.08162},
    archivePrefix = {arXiv},
    primaryClass = {q-bio.BM}
}
```

```bibtex
@misc{alquraishi2019proteinnet,
    title   = {ProteinNet: a standardized data set for machine learning of protein structure}, 
    author  = {Mohammed AlQuraishi},
    year    = {2019},
    eprint  = {1902.00249},
    archivePrefix = {arXiv},
    primaryClass = {q-bio.BM}
}
```

```bibtex
@misc{gomez2017reversible,
    title     = {The Reversible Residual Network: Backpropagation Without Storing Activations}, 
    author    = {Aidan N. Gomez and Mengye Ren and Raquel Urtasun and Roger B. Grosse},
    year      = {2017},
    eprint    = {1707.04585},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{fuchs2021iterative,
    title   = {Iterative SE(3)-Transformers},
    author  = {Fabian B. Fuchs and Edward Wagstaff and Justas Dauparas and Ingmar Posner},
    year    = {2021},
    eprint  = {2102.13419},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@misc{satorras2021en,
    title   = {E(n) Equivariant Graph Neural Networks}, 
    author  = {Victor Garcia Satorras and Emiel Hoogeboom and Max Welling},
    year    = {2021},
    eprint  = {2102.09844},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
