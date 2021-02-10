<img src="./images/alphafold2.png" width="600px"></img>

## Alphafold2 - Pytorch (wip)

To eventually become an unofficial working Pytorch implementation of <a href="https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology">Alphafold2</a>, the breathtaking attention network that solved CASP14. Will be gradually implemented as more details of the architecture is released.

Once this is replicated, I intend to fold all available amino acid sequences out there in-silico and release it as an academic torrent, to further science. If you are interested in replication efforts, please drop by #alphafold at this <a href="https://discord.com/invite/vtRgjbM">Discord channel</a>

## Install

```bash
$ pip install alphafold2-pytorch
```

## Usage

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

distances, weights = center_distogram_torch(distogram)

coords_3d, _ = MDScaling(distances, 
    weights = weights,
    iters = 200, 
    fix_mirror = 0
)
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

## Equivariant Attention

There are two equivariant self attention libraries that I have prepared for the purposes of replication. One is the implementation by Fabian Fuchs as detailed in a <a href="https://fabianfuchsml.github.io/alphafold2/">speculatory blogpost</a>. The other is from a recent paper from Deepmind, claiming their approach is better than using irreducible representations.

- <a href="https://github.com/lucidrains/se3-transformer-pytorch">SE3 Transformer</a>
- <a href="https://github.com/lucidrains/lie-transformer-pytorch">Lie Transformer</a>

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
