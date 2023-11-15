import os
import torch
from tqdm import tqdm
from torch.nn.functional import pad
import torchtext.datasets as datasets
from sklearn.model_selection import train_test_split
from torchtext.data.functional import to_map_style_dataset

class SentenceProcessor:
    @classmethod
    def sentence_to_tokens(cls, sentence, tokenizer, vocab, max_padding):
        '''
        Performs the following operations:
        - tokenize sentence
        - convert data type to tensor
        - add beginning of sentence and end of sentence markers
        - pad end of sentence up to the max_padding argument
        '''
        # get ids for beginning and end of sentence, and padding markers
        bos_id, eos_id, pad_id = vocab(["<s>", "</s>", "<blank>"])
        # tokenize sentence into individual tokens
        tokens = tokenizer.tokenize(sentence)
        # map tokens to ids and add beginning, end of sentence markers
        token_ids = [bos_id] + vocab(tokens) + [eos_id]
        # convert to tensor for compatibility with padding function
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.int64)
        # pad token ids
        pad_size = (0, max_padding - len(token_ids_tensor))
        padded_token_ids_tensor = pad(token_ids_tensor, pad_size, value=pad_id)
        return padded_token_ids_tensor
    
    @classmethod
    def tokens_to_sentence(cls, padded_token_ids_tensor, vocab):
        '''
        Reverse operations to those described under 
        :meth:`<processors.SentenceProcessor.sentence_to_tokens>`
        Note that we choose to remove bos and eos tokens
        '''
        # get ids for beginning and end of sentence, and padding markers
        bos_id, eos_id, pad_id = vocab(["<s>", "</s>", "<blank>"])
        # convert to list
        padded_token_ids = padded_token_ids_tensor.tolist()
        # remove padding
        unpadded_token_ids = [tok for tok in padded_token_ids if tok != pad_id]
        # stop at the first occurence of end of sentence token
        eos_position = unpadded_token_ids.index(eos_id) if eos_id in unpadded_token_ids else len(unpadded_token_ids)
        token_ids = unpadded_token_ids[1 : eos_position]
        # get tokens from token ids
        tokens = vocab.lookup_tokens(token_ids)
        sentence = " ".join(tokens)
        return sentence
    

class DataProcessor(SentenceProcessor):
    def __init__(self, tokenizer_src, tokenizer_tgt, vocab_src, vocab_tgt,
                 max_padding, language_pair):
        super().__init__()
        # TODO: language_pair arg comes first for consistency
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_padding = max_padding
        self.language_pair = language_pair

    def preprocess_data(self, raw_data):
        '''
        Preprocess raw data sentence by sentence, as detailed in 
        :meth:`<processors.SentenceProcessor.sentence_to_tokens>`
        '''
        data_map = to_map_style_dataset(raw_data) # map allows access to length of dataset
        # initialize an empty preprocessed dataset
        preprocessed_dataset = torch.zeros(
            size=(len(data_map), len(data_map[0]), self.max_padding),
            dtype=torch.int64,
        )
        # populate the preprocessed dataset, sentence pair by sentence pair
        for i, (src, tgt) in enumerate(tqdm(data_map, desc="Preprocessing dataset")):
            preprocessed_src = self.sentence_to_tokens(src, 
                                                       self.tokenizer_src,
                                                       self.vocab_src,
                                                       self.max_padding)
            preprocessed_tgt = self.sentence_to_tokens(tgt, 
                                                       self.tokenizer_tgt,
                                                       self.vocab_tgt,
                                                       self.max_padding)
            preprocessed_dataset[i][0] = preprocessed_src
            preprocessed_dataset[i][1] = preprocessed_tgt
        
        return preprocessed_dataset
    
    def get_preprocessed_data(self):
        '''
        Load the preprocessed data if saved, else download and 
        preprocess data
        '''
        save_path = "artifacts/saved_data/preprocd_data.pt"
        if os.path.exists(save_path):
            preprocd_data = torch.load(save_path)
        else:
            raw_data = self.get_raw_data(self.language_pair)
            preprocd_data = self.preprocess_data(raw_data)
            torch.save(preprocd_data, save_path)
        return preprocd_data
        
    # TODO: make this the standard function
    # def get_data(self, option):
    #     '''
    #     Gets data from the Pytorch Multi30k dataset. Takes two options, 'raw'
    #     and 'preprocessed' and returns a dataset in the corresponding format:
    #     - 'raw': iterator of 2-tuples, each containing a source sentence and 
    #        a target sentence in string format
    #     - 'preprocessed': tensor of shape [num_sentences, max_sentence_length, 2]
    #        where 2 denotes source and target language sentences. Note that the 
    #        sentences are preprocessed and tokenized as described under 
    #        :meth:`<processors.DataProcessor.preprocess_data>`
    #     '''
    #     # get raw data
    #     train_iter, valid_iter, test_iter = datasets...(...)
    #     raw_data = train_iter + valid_iter + test_iter
    #     if option == 'raw':
    #         return raw_data
    #     elif option == 'preprocessed':

    @staticmethod
    def get_raw_data(language_pair):
        train_iter, valid_iter, test_iter = datasets.Multi30k(language_pair=language_pair) #TODO: leverage split argument
        # train_iter, valid_iter, test_iter = datasets.IWSLT2017(language_pair=language_pair)
        raw_data = train_iter + valid_iter + test_iter
        return raw_data
    
    @staticmethod
    def get_data_splits(data, split_ratio=(0.8, 0.1, 0.1), random_seed=None):
        '''
        Splits a given dataset into train, validation and test sets as
        determined by the specified split ratio
        TODO: deprecate this if split is done in get_raw_data
        '''
        train_size, val_size, test_size = split_ratio
        train_data, val_and_test_data = train_test_split(
            data,
            train_size=train_size,
            random_state=random_seed
        )
        val_data, test_data = train_test_split(
            val_and_test_data,
            train_size=val_size/(val_size+test_size),
            random_state=random_seed
        )
        return train_data, val_data, test_data