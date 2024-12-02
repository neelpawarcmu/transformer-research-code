import torch
import spacy
import os
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.functional import pad
from data.sources import DataSource
 
class VocabularyBuilder:
    def __init__(self, tokenize_fn, data, language_type):
        self.language_type = language_type
        self.tokenize_fn = tokenize_fn
        self.vocab = self.build_vocabulary(data)
        self.vocab.length = len(self.vocab)

    def build_vocabulary(self, data):
        '''
        Creates vocab from tokenized raw data
        '''
        vocab = build_vocab_from_iterator(
            iterator = self._yield_tokens(data_iterator = data),
            min_freq = 2,
            specials = ["<s>", "</s>", "[PAD]", "[UNK]"],
        )
        vocab.set_default_index(vocab["[UNK]"])
        return vocab
    
    def _yield_tokens(self, data_iterator):
        for src_sentence, tgt_sentence in data_iterator:
            if self.language_type == "src":
                yield self.tokenize_fn(src_sentence)
            elif self.language_type == "tgt":
                yield self.tokenize_fn(tgt_sentence)

class SpacyTokenizer:
    def __init__(self, language, language_type, language_pair):
        self.available_pipelines = {"en": "en_core_web_sm", 
                                    "de": "de_core_news_sm",
                                    "ro": "ro_core_news_md"}
        self.language = language
        self.language_type = language_type
        self.language_pair = language_pair
        self.load_spacy_pipeline()
        self.bos_token, self.bos_token_id = "<s>", 0
        self.eos_token, self.eos_token_id = "</s>", 1
        self.pad_token, self.pad_token_id = "[PAD]", 2

    def load_spacy_pipeline(self):
        pipeline_name = self.available_pipelines[self.language]
        try:
            self.spacy_pipeline = spacy.load(pipeline_name)
        except:
            os.system(f"python3 -m spacy download {pipeline_name}")
            self.spacy_pipeline = spacy.load(pipeline_name)

    def tokenize_sentence(self, sentence):
        words = [tok.text for tok in self.spacy_pipeline.tokenizer(sentence)]
        return words

    def build_vocabulary(self, dataset_name):
        raw_data = DataSource.get_data(name=dataset_name, 
                                       language_pair=self.language_pair, 
                                       cache=True)
        self.vocab = VocabularyBuilder(self.tokenize_sentence, 
                                       raw_data,
                                       self.language_type).vocab

    def __call__(self, sentences, padding=True, truncation=True, max_length=None):
        print("Call to spacy tokenizer")
        pbar = tqdm(sentences, desc="Tokenizing sentences")
        tokens = [self.sentence_to_tokens(sentence, max_length) for sentence in pbar]
        token_dict = {"input_ids": tokens} # for compatibility with the hf tokenizer scripts
        return token_dict

    def batch_decode(self, batch_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        pbar = tqdm(batch_token_ids, desc="Decoding sentences", leave=False)
        return [self.tokens_to_sentence(token_ids) for token_ids in pbar]
    
    def sentence_to_tokens(self, sentence, max_padding):
        '''
        Performs the following operations:
        - tokenize sentence
        - convert data type to tensor
        - add beginning of sentence and end of sentence markers
        - pad end of sentence up to the max_padding argument
        '''
        # get ids for beginning and end of sentence, and padding markers
        bos_id, eos_id, pad_id = self.bos_token_id, self.eos_token_id, self.pad_token_id
        # tokenize sentence into individual tokens
        tokens = self.tokenize_sentence(sentence)
        # map tokens to ids and add beginning, end of sentence markers
        token_ids = [bos_id] + self.vocab(tokens) + [eos_id]
        # convert to tensor for compatibility with padding function
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.int64)
        # pad token ids
        pad_size = (0, max_padding - len(token_ids_tensor))
        padded_token_ids_tensor = pad(token_ids_tensor, pad_size, value=pad_id)
        return padded_token_ids_tensor.tolist() # this is for compatibility with the hf tokenizer scripts
    
    def tokens_to_sentence(self, padded_token_ids_tensor):
        '''
        Reverse operations to those described under 
        :meth:`<processors.SentenceProcessor.sentence_to_tokens>`
        Note that we choose to remove bos and eos tokens
        '''
        # get ids for beginning and end of sentence, and padding markers
        bos_id, eos_id, pad_id = self.vocab(["<s>", "</s>", "<blank>"])
        # convert to list
        padded_token_ids = padded_token_ids_tensor.tolist()
        # remove padding
        unpadded_token_ids = [tok for tok in padded_token_ids if tok != pad_id]
        # stop at the first occurence of end of sentence token
        eos_position = unpadded_token_ids.index(eos_id) if eos_id in unpadded_token_ids else len(unpadded_token_ids)
        token_ids = unpadded_token_ids[0 : eos_position]
        # get tokens from token ids
        tokens = self.vocab.lookup_tokens(token_ids)
        sentence = " ".join(tokens)
        return sentence



# def load_vocabularies(tokenizer_src=None, tokenizer_tgt=None, data=None):
#     """
#     Loads vocabs if saved, else creates and saves them locally.
#     """
#     vocab_src = VocabularyBuilder(tokenizer_src, data, "src").vocab
#     vocab_tgt = VocabularyBuilder(tokenizer_tgt, data, "tgt").vocab
    
#     print("-"*80)
#     print(f"{vocab_src.language.upper()} vocabulary size: {len(tokenizer_src.vocab)}")
#     print(f"{vocab_tgt.language.upper()} vocabulary size: {len(tokenizer_tgt.vocab)}")
#     print("-"*80)
#     return vocab_src, vocab_tgt
