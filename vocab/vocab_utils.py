import torch
import spacy
import os
from torchtext.vocab import build_vocab_from_iterator

class VocabularyBuilder:
    def __init__(self, tokenizer, data, vocab_type):
        self.vocab_type = vocab_type
        self.tokenize_fn = tokenizer.tokenize
        self.vocab = self.build_vocabulary(data)
        self.vocab.length = len(self.vocab)
        self.vocab.language = tokenizer.language

    def build_vocabulary(self, data):
        '''
        Creates vocab from tokenized raw data
        '''
        vocab = build_vocab_from_iterator(
            iterator = self._yield_tokens(data_iterator = data),
            min_freq = 2,
            specials = ["<s>", "</s>", "<blank>", "<unk>"],
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab
    
    def _yield_tokens(self, data_iterator):
        for src_sentence, tgt_sentence in data_iterator:
            if self.vocab_type == "src":
                yield self.tokenize_fn(src_sentence)
            elif self.vocab_type == "tgt":
                yield self.tokenize_fn(tgt_sentence)

class SpacyTokenizer:
    def __init__(self, language):
        self.available_pipelines = {"en": "en_core_web_sm", 
                                    "de": "de_core_news_sm",
                                    "ro": "ro_core_news_md"}
        self.language = language
        self.load_spacy_pipeline()
       
    def load_spacy_pipeline(self):
        pipeline_name = self.available_pipelines[self.language]
        try:
            self.spacy_pipeline = spacy.load(pipeline_name)
        except:
            os.system(f"python3 -m spacy download {pipeline_name}")
            self.spacy_pipeline = spacy.load(pipeline_name)

    def tokenize(self, text):
        return [tok.text for tok in self.spacy_pipeline.tokenizer(text)]

# build tokenizers and vocabularies for source and target languages
def build_tokenizers(language_pair):
    tokenizer_src = SpacyTokenizer(language_pair[0])
    tokenizer_tgt = SpacyTokenizer(language_pair[1])
    return tokenizer_src, tokenizer_tgt

def load_vocabularies(tokenizer_src=None, tokenizer_tgt=None, data=None, cache=False):
    """
    Loads vocabs if saved, else creates and saves them locally.
    """
    cache_path = "artifacts/saved_vocab/vocabs.pt"

    if cache and os.path.exists(cache_path):
        print(f"Loaded vocab from {cache_path}")
        vocab_src, vocab_tgt = torch.load(cache_path)
    else:
        vocab_src = VocabularyBuilder(tokenizer_src, data, "src").vocab
        vocab_tgt = VocabularyBuilder(tokenizer_tgt, data, "tgt").vocab
        torch.save((vocab_src, vocab_tgt), cache_path)
    
    print("-"*80)
    print(f"{vocab_src.language.upper()} vocabulary size: {vocab_src.length}")
    print(f"{vocab_tgt.language.upper()} vocabulary size: {vocab_tgt.length}")
    print("-"*80)
    return vocab_src, vocab_tgt
