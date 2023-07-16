import torch
import spacy
import os
from os.path import exists
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torch.nn.functional import pad

# "Batching"

def get_subseq_tokens_mask(tgt):
    """
    Generate an upper triangular matrix to mask out subsequent positions
    """
    mask_shape = (1, tgt.size(-1), tgt.size(-1))
    upper_tri_matrix = torch.triu(torch.ones(mask_shape), diagonal=1)
    subseq_tokens_mask = (upper_tri_matrix == 0).type_as(tgt)
    return subseq_tokens_mask

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad_idx=2):  # 2 = <blank>
        self.src = src
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.decoder_attn_mask = self.make_decoder_attn_mask(self.tgt, pad_idx)
            self.ntokens = (self.tgt_y != pad_idx).data.sum()

    @staticmethod
    def make_decoder_attn_mask(tgt, pad_idx):
        "Create a mask to hide padding and future words."
        pad_mask = (tgt != pad_idx).unsqueeze(-2)
        subseq_tokens_mask = get_subseq_tokens_mask(tgt)
        decoder_attn_mask = pad_mask & subseq_tokens_mask
        return decoder_attn_mask

# "Load vocab and tokenizers"

def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except:
        os.system("python3 -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except:
        os.system("python3 -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")
    return spacy_de, spacy_en

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("artifacts/saved_vocab/vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "artifacts/saved_vocab/vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("artifacts/saved_vocab/vocab.pt")
    print(f"Finished loading vocabulary.")
    print(f"German vocabulary size: {len(vocab_src)}")
    print(f"English vocabulary size: {len(vocab_tgt)}")
    return vocab_src, vocab_tgt

# "Iterators"

def collate_batch(
    batch,
    tokenize_de,
    tokenize_en,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(tokenize_de(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tokenize_en(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

def create_dataloaders(device, vocab_src, vocab_tgt, spacy_de, 
                       spacy_en, shuffle=True, batch_size=12000, 
                       max_padding=128):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_map = to_map_style_dataset(train_iter)
    valid_map = to_map_style_dataset(valid_iter)

    train_dataloader = DataLoader(
        train_map,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_map,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader