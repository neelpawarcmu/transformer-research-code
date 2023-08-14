import torch
import spacy
import os
from os.path import exists
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torch.nn.functional import pad
from visualizations.viz import visualize_attn_mask

# "Batching"
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad_idx=2):  # 2 = <blank>
        self.src = src
        if tgt is not None:
            self.tgt_shifted_right = tgt[:, :-1] # everything except last pad token
            self.tgt_label = tgt[:, 1:] # everything except beginning of sentence token
            self.decoder_attn_mask = self.make_decoder_attn_mask(
                self.tgt_shifted_right, pad_idx
            )
            self.ntokens = (self.tgt_label != pad_idx).sum()

    @staticmethod
    def make_decoder_attn_mask(tgt, pad_idx):
        "Create a mask to hide padding and future words."
        pad_mask = (tgt != pad_idx).unsqueeze(-2)
        pad_mask_T = pad_mask.transpose(1,2)
        subseq_tokens_mask = Batch.get_subseq_tokens_mask(tgt)
        decoder_attn_mask = pad_mask & subseq_tokens_mask & pad_mask_T
        # TODO remove
        # visualize_attn_mask(decoder_attn_mask)
        # import pdb; pdb.set_trace()
        return decoder_attn_mask
    
    @staticmethod
    def get_subseq_tokens_mask(tgt):
        """
        Generate an upper triangular matrix to mask out subsequent positions
        """
        mask_shape = (1, tgt.size(-1), tgt.size(-1))
        upper_tri_matrix = torch.triu(torch.ones(mask_shape), diagonal=1)
        subseq_tokens_mask = (upper_tri_matrix == 0).type_as(tgt)
        return subseq_tokens_mask