import torch
import spacy
import os
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets

"""
class DataLoader

class Dataset

class Tokenizer

class Vocab
"""

class Vocabulary:
    def __init__(self, language, tokenizer):
        self.language = language
        self.tokenize_fn = tokenizer.tokenize
        self.build_vocabulary()

    def build_vocabulary(self):
        # get raw data
        train, val, test = datasets.Multi30k(language_pair=("de", "en"))

        # create vocab from tokenized raw data
        vocab = build_vocab_from_iterator(
            iterator = self._yield_tokens(data_iterator = train+val+test),
            min_freq = 2,
            specials = ["<s>", "</s>", "<blank>", "<unk>"],
        )
        vocab.set_default_index(vocab["<unk>"])
        self.vocab = vocab
    
    def _yield_tokens(self, data_iterator):
        for ger, eng in data_iterator:
            if self.language == "german":
                yield self.tokenize_fn(ger)
            elif self.language == "english":
                yield self.tokenize_fn(eng)

class SpacyTokenizer:
    def __init__(self, language):
        self.model_names = {"english": "en_core_web_sm", 
                            "german": "de_core_news_sm"}
        self.language = language
        self.load_spacy_model()
       
    def load_spacy_model(self):
        model_name = self.model_names[self.language]
        try:
            self.spacy_model = spacy.load(model_name)
        except:
            os.system(f"python3 -m spacy download {model_name}")
            self.spacy_model = spacy.load(model_name)

    def tokenize(self, text):
        return [tok.text for tok in self.spacy_model.tokenizer(text)]

# build tokenizers and vocabularies for source and target languages
def build_tokenizers():
    tokenizer_ger = SpacyTokenizer("german")
    tokenizer_eng = SpacyTokenizer("english")
    return tokenizer_ger, tokenizer_eng

def build_vocabularies(spacy_tokenizer_ger, spacy_tokenizer_eng):
    """
    Load vocabs if saved, else create and save them locally.
    """
    save_path = "artifacts/saved_vocab/vocabs.pt"

    if os.path.exists(save_path):
        vocab_ger, vocab_eng = torch.load(save_path)
    else:
        vocab_ger = Vocabulary("german", spacy_tokenizer_ger)
        vocab_eng = Vocabulary("english", spacy_tokenizer_eng)
        torch.save((vocab_ger, vocab_eng), save_path)
    
    print("-"*80)
    print(f"German vocabulary size: {len(vocab_ger.vocab)}")
    print(f"English vocabulary size: {len(vocab_eng.vocab)}")
    print("-"*80)
    return vocab_ger, vocab_eng
