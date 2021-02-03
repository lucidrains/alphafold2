<img src="./images/alphafold2.png" width="600px"></img>

## Alphafold2 - Pytorch (wip)

To eventually become an unofficial working Pytorch implementation of <a href="https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology">Alphafold2</a>, the breathtaking attention network that solved CASP14. Will be gradually implemented as more details of the architecture is released.

If you are interested in replication efforts, please drop by #alphafold at this <a href="https://discord.com/invite/vtRgjbM">Discord channel</a>

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
    dim_head = 64
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

## Testing

```bash
$ python setup.py test
```

## Data

This library will use the awesome work by <a href="http://github.com/jonathanking">Jonathan King</a> at <a href="https://github.com/jonathanking/sidechainnet">this repository</a>.

To install

```bash
$ git clone https://github.com/jonathanking/sidechainnet.git
$ cd sidechainnet && pip install -e .
```

## Related repositories

I have started construction of two approaches to equivariant self-attention. Both are still works in progress but should be done by end of January. Feel free to contribute.

Update - SE3 Transformers is in a good place, but Lie Transformer could use a code review, specifically how the location attention is handled

- [ ] https://github.com/lucidrains/lie-transformer-pytorch
- [x] https://github.com/lucidrains/se3-transformer-pytorch

## Speculation

https://xukui.cn/alphafold2.html

https://fabianfuchsml.github.io/alphafold2/

https://moalquraishi.wordpress.com/2020/12/08/alphafold2-casp14-it-feels-like-ones-child-has-left-home/

<img src="./images/science.png"></img>

<img src="./images/reddit.png"></img>

Developments from competing labs

https://www.biorxiv.org/content/10.1101/2020.12.10.419994v1.full.pdf

# External packages

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
