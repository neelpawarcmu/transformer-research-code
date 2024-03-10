import torch
import os.path as op
from transformers import AutoTokenizer

def build_tokenizers(language_pair, cache=True):
    name = "bert"
    cache_path = f"artifacts/saved_tokenizers/{name}_tokenizers.pt"
    if (cache and op.exists(cache_path)):
        tokenizer_src, tokenizer_tgt = torch.load(cache_path)
        print(f"Loaded tokenizers from {cache_path}")
    else:
        available_tokenizers = {"en": "bert-base-cased",
                                "de": "dbmdz/bert-base-german-cased"}
        tokenizer_src = AutoTokenizer.from_pretrained(available_tokenizers[language_pair[0]])
        tokenizer_tgt = AutoTokenizer.from_pretrained(available_tokenizers[language_pair[1]])
        tokenizer_src.add_special_tokens({"bos_token":"<s>",
                                        "eos_token":"</s>"})
        tokenizer_tgt.add_special_tokens({"bos_token":"<s>",
                                        "eos_token":"</s>"})
        torch.save((tokenizer_src, tokenizer_tgt), cache_path)
    return tokenizer_src, tokenizer_tgt