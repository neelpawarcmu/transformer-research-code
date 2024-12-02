import os
import spacy
import torch
import os.path as op
from transformers import AutoTokenizer
from tokenization.spacy_tokenizer_utils import SpacyTokenizer
from utils.config import Config

def build_tokenizers(config: Config):
    language_pair = config.dataset.language_pair
    tokenizer_type = "spacy" if config.harvard else config.tokenizer.type
    cache = config.tokenizer.cache  
    cache_path = op.join(config.logging.artifacts_dir, 
                         config.logging.subdirs.tokenizers, 
                         f"{tokenizer_type}_tokenizers.pt")
    if (cache and op.exists(cache_path)):
        tokenizer_src, tokenizer_tgt = torch.load(cache_path)
        print(f"Loaded saved tokenizers from {cache_path}")
    else:
        if tokenizer_type == "spacy":
            tokenizer_src, tokenizer_tgt = build_spacy_tokenizers(language_pair)
            dataset_name = "m30k" # HARVARD
            tokenizer_src.build_vocabulary(dataset_name)
            tokenizer_tgt.build_vocabulary(dataset_name)
        else:
            tokenizer_src, tokenizer_tgt = build_bert_tokenizers(language_pair)
        torch.save((tokenizer_src, tokenizer_tgt), cache_path)
        print(f"Saved tokenizers to {cache_path}")
    print(f"Src vocab size: {len(tokenizer_src.vocab)}")
    print(f"Tgt vocab size: {len(tokenizer_tgt.vocab)}")
    return tokenizer_src, tokenizer_tgt

def build_bert_tokenizers(language_pair):
    available_tokenizers = {"en": "bert-base-cased",
                            "de": "dbmdz/bert-base-german-cased"}
    tokenizer_src = AutoTokenizer.from_pretrained(available_tokenizers[language_pair[0]])
    tokenizer_tgt = AutoTokenizer.from_pretrained(available_tokenizers[language_pair[1]])
    return tokenizer_src, tokenizer_tgt

# build tokenizers and vocabularies for source and target languages
def build_spacy_tokenizers(language_pair):
    tokenizer_src = SpacyTokenizer(language_pair[0], "src", language_pair)
    tokenizer_tgt = SpacyTokenizer(language_pair[1], "tgt", language_pair)
    return tokenizer_src, tokenizer_tgt