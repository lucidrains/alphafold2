import torch
import torch.nn.functional as F
from torch import nn

from alphafold2_pytorch.utils import get_msa_embedd, get_esm_embedd, exists
from alphafold2_pytorch.constants import MSA_MODEL_PATH, MSA_EMBED_DIM, ESM_MODEL_PATH, ESM_EMBED_DIM

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

    def forward(self, seq, msa, msa_mask = None, **kwargs):
        assert seq.shape[-1] == msa.shape[-1], 'sequence and msa must have the same length if you wish to use MSA transformer embeddings'
        model, batch_converter, device = self.model, self.batch_converter, seq.device

        seq_and_msa = torch.cat((seq.unsqueeze(1), msa), dim = 1)

        if exists(msa_mask):
            # in the event that there are rows in the MSA that are completely padding
            # process each batch element individually, so that padding isn't processed
            # with row-tied attention

            num_msa = msa_mask.any(dim = -1).sum(dim = -1).tolist()
            seq_and_msa_list = seq_and_msa.unbind(dim = 0)
            num_rows = seq_and_msa.shape[1]

            embeds = []
            for num, batch_el in zip(num_msa, seq_and_msa_list):
                batch_el = rearrange(batch_el, '... -> () ...')
                batch_el = batch_el[:, :num]
                embed = get_msa_embedd(batch_el, model, batch_converter, device = device)
                embed = F.pad(embed, (0, 0, 0, 0, 0, num_rows - num), value = 0.)
                embeds.append(embed)

            embeds = torch.cat(embeds, dim = 0)
        else:
            embeds = get_msa_embedd(seq_and_msa, model, batch_converter, device = device)

        embeds = self.project_embed(embeds)
        seq_embed, msa_embed = embeds[:, 0], embeds[:, 1:]

        return self.alphafold2(seq, msa, seq_embed = seq_embed, msa_embed = msa_embed, msa_mask = msa_mask, **kwargs)

class ESMEmbedWrapper(nn.Module):
    def __init__(self, *, alphafold2):
        super().__init__()
        self.alphafold2 = alphafold2

        model, alphabet = torch.hub.load(*ESM_MODEL_PATH) 
        batch_converter = alphabet.get_batch_converter()

        self.model = model
        self.batch_converter = batch_converter
        self.project_embed = nn.Linear(ESM_EMBED_DIM, alphafold2.dim) if ESM_EMBED_DIM != alphafold2.dim else nn.Identity()

    def forward(self, seq, msa=None, **kwargs):
        model, batch_converter, device = self.model, self.batch_converter, seq.device

        seq_embeds = get_esm_embedd(seq, model, batch_converter, device = device)
        seq_embeds = self.project_embed(seq_embeds)

        if msa is not None:
            flat_msa = rearrange(msa, 'b m n -> (b m) n')
            msa_embeds = get_esm_embedd(flat_msa, model, batch_converter, device = device)
            msa_embeds = rearrange(msa_embeds, '(b m) n d -> b m n d')
            msa_embeds = self.project_embed(msa_embeds)
        else: 
            msa_embeds = None

        return self.alphafold2(seq, msa, seq_embed = seq_embeds, msa_embed = msa_embeds, **kwargs)
