import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from alphafold2_pytorch import constants
from einops import rearrange

# MSA MLM

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

class MLM(nn.Module):
    def __init__(
        self,
        dim,
        num_tokens,
        mask_id,
        mask_prob = 0.15,
        random_replace_token_prob = 0.1,
        keep_token_same_prob = 0.1,
        exclude_token_ids = (0,)
    ):
        super().__init__()
        self.to_logits = nn.Linear(dim, num_tokens)
        self.mask_id = mask_id

        self.mask_prob = mask_prob
        self.exclude_token_ids = exclude_token_ids
        self.keep_token_same_prob = keep_token_same_prob
        self.random_replace_token_prob = random_replace_token_prob

    def noise(self, seq, mask):
        num_msa = seq.shape[1]
        seq = rearrange(seq, 'b n ... -> (b n) ...')
        mask = rearrange(mask, 'b n ... -> (b n) ...')

        # prepare masks for noising sequence

        excluded_tokens_mask = mask

        for token_id in self.exclude_token_ids:
            excluded_tokens_mask = excluded_tokens_mask & (seq != token_id)

        mlm_mask = get_mask_subset_with_prob(excluded_tokens_mask, self.mask_prob)

        # keep some tokens the same

        replace_token_with_mask = get_mask_subset_with_prob(mlm_mask, 1. - self.keep_token_same_prob)

        # replace with mask

        seq = seq.masked_fill(mlm_mask, self.mask_id)

        # generate random tokens

        random_replace_token_prob_mask = get_mask_subset_with_prob(mlm_mask, (1 - self.keep_token_same_prob) * self.random_replace_token_prob)

        random_tokens = torch.randint(1, constants.NUM_AMINO_ACIDS, seq.shape).to(seq.device)

        for token_id in self.exclude_token_ids:
            random_replace_token_prob_mask = random_replace_token_prob_mask & (random_tokens != token_id)  # make sure you never substitute a token with an excluded token type (pad, start, end)

        # noise sequence

        noised_seq = torch.where(random_replace_token_prob_mask, random_tokens, seq)
        noised_seq = rearrange(noised_seq, '(b n) ... -> b n ...', n = num_msa)
        mlm_mask = rearrange(mlm_mask, '(b n) ... -> b n ...', n = num_msa)

        return noised_seq, mlm_mask

    def forward(self, seq_embed, original_seq, mask):
        logits = self.to_logits(seq_embed)
        seq_logits = logits[mask]
        seq_labels = original_seq[mask]

        loss = F.cross_entropy(seq_logits, seq_labels, reduction = 'mean')
        return loss
