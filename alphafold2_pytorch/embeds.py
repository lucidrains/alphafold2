import torch
from torch import nn

from alphafold2_pytorch.utils import get_msa_embedd, get_esm_embedd
from alphafold2_pytorch.constants import MSA_MODEL_PATH, MSA_EMBED_DIM

from einops import rearrange

class MSAEmbedWrapper(nn.Module):
    def __init__(self, *, alphafold2):
        super().__init__()
        self.alphafold2 = alphafold2

        model, alphabet = torch.hub.load(*MSA_MODEL_PATH) 
        batch_converter = alphabet.get_batch_converter()

        self.model = model
        self.batch_converter = batch_converter
        self.project_embed = nn.Linear(MSA_EMBED_DIM, alphafold2.dim) if MSA_EMBED_DIM != alphafold2.dim else nn.Identity()

    def forward(self, seq, msa, **kwargs):
        assert seq.shape[-1] == msa.shape[-1], 'sequence and msa must have the same length if you wish to use MSA transformer embeddings'
        model, batch_converter, device = self.model, self.batch_converter, seq.device

        seq_and_msa = torch.cat((seq.unsqueeze(1), msa), dim = 1)
        embeds = get_msa_embedd(seq_and_msa, model, batch_converter, device = device)
        embeds = self.project_embed(embeds)

        seq_embed, msa_embed = embeds[:, 0], embeds[:, 1:]

        return self.alphafold2(seq, msa, seq_embed = seq_embed, msa_embed = msa_embed, **kwargs)

class ESMEmbedWrapper(nn.Module):
    def __init__(self, *, alphafold2):
        super().__init__()
        self.alphafold2 = alphafold2

        model, alphabet = torch.hub.load(*ESM_MODEL_PATH) 
        batch_converter = alphabet.get_batch_converter()

        self.model = model
        self.batch_converter = batch_converter
        self.project_embed = nn.Linear(ESM_EMBED_DIM, alphafold2.dim) if ESM_EMBED_DIM != alphafold2.dim else nn.Identity()

    def forward(self, seq, msa, **kwargs):
        model, batch_converter, device = self.model, self.batch_converter, seq.device
        num_msa = msa.shape[1]

        seq_embeds = get_esm_embedd(seq, model, batch_converter, device = device)

        flat_msa = rearrange(msa, 'b m n -> (b m) n')
        msa_embeds = get_esm_embedd(flat_msa, model, batch_converter, device = device)
        msa_embeds = rearrange(msa_embeds, '(b m) n d -> b m n d')

        seq_embeds, msa_embeds = map(lambda t: self.project_embed(t), (seq_embeds, msa_embeds))
        return self.alphafold2(seq, msa, seq_embed = seq_embed, msa_embed = msa_embed, **kwargs)
