import torch
import os.path as op
import torchtext.datasets as datasets
from datasets import load_dataset # TODO: move to download.py

class DataDownloader:
    @staticmethod
    def get_data(name, language_pair, cache, preprocess, preprocessor=None):
        '''
        Gets data from the Pytorch Multi30k dataset. Returns data in one of two
        formats, 'raw' or 'preprocessed':
        - 'raw': iterator of length num_sentences and containing 2-tuples, 
           each containing a source sentence and a target sentence as strings
        - 'preprocessed': tensor of shape [num_sentences, max_sentence_length, 2]
           where 2 denotes source and target language sentences. Note that the 
           sentences are preprocessed and tokenized as described under 
           :meth:`<processors.DataProcessor.preprocess_data>`
        Args: 
        name: Name of dataset, options: ['wmt14', 'm30k']
        '''
        # conditionally return saved data directly
        cache_path = "artifacts/saved_data/preprocd_data.pt"
        if (preprocess and cache and op.exists(cache_path)):
            preprocessed_data = torch.load(cache_path)
            print(f"Loaded data from {cache_path}")
            return preprocessed_data

        # get raw data
        if name == 'wmt14':
            dataset_dict = load_dataset("wmt14", "-".join(language_pair))
            train, val, test = (dataset_dict['train']['translation'], 
                                dataset_dict['validation']['translation'], 
                                dataset_dict['test']['translation'])
            raw_data = [tuple(sentence_pair.values()) for sentence_pair in train + val + test]
        elif name == 'm30k':
            train_iter, valid_iter, test_iter = datasets.Multi30k(language_pair=language_pair) 
            raw_data = train_iter + valid_iter + test_iter
        else: 
            raise ValueError(f"Received {name}, available datasets 'wmt14' and 'm30k'")
        
        # preprocess and save if needed
        if preprocess:
            preprocessed_data = preprocessor.preprocess_data(raw_data)
            torch.save(preprocessed_data, cache_path) 
            return preprocessed_data
        else:
            return raw_data