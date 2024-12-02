import torch
import os.path as op
from datasets import load_dataset # huggingface datasets
from data.preprocess import DataProcessor

class DataSource:
    @classmethod
    def get_data(cls, name, language_pair, cache=False, preprocessor=None, dataset_size=5000000, max_padding=60, random_seed=None):
        '''
        Returns the raw or preprocessed data for a given dataset.
        '''
        if preprocessor:
            data = cls._get_preprocessed_data(name, language_pair, cache, preprocessor, random_seed)
        else:
            data = cls._get_raw_data(name, language_pair, cache, random_seed)
        data = DataProcessor.limit_train_size(data, dataset_size, max_padding)
        return data

    @classmethod
    def _get_raw_data(cls, name, language_pair, cache, random_seed=None, dataset_path="artifacts/saved_data/de_en.txt"):
        # conditionally return saved data directly
        cache_path = f"artifacts/saved_data/raw_data/{name}.pt"
        if (cache and op.exists(cache_path)):
            raw_data = torch.load(cache_path)
            print(f"Loaded raw data from {cache_path}")
        else:
            raw_data = DataDownloader.download(name, language_pair, random_seed, dataset_path)
            torch.save(raw_data, cache_path)
        return raw_data

    @classmethod
    def _get_preprocessed_data(cls, name, language_pair, cache, preprocessor, random_seed=None, dataset_path="artifacts/saved_data/de_en.txt"):
        # conditionally return saved data directly
        cache_path = f"artifacts/saved_data/preprocessed_data/{name}.pt"
        if (cache and op.exists(cache_path)):
            preprocessed_data = torch.load(cache_path)
            print(f"Loaded preprocessed data from {cache_path}")
        else:
            raw_data = cls._get_raw_data(name, language_pair, cache, random_seed, dataset_path)
            preprocessed_data = preprocessor.preprocess_data(raw_data)
            torch.save(preprocessed_data, cache_path)
        return preprocessed_data


class DataDownloader:
    @staticmethod
    def download(name, language_pair, random_seed=None, dataset_path="artifacts/saved_data/de_en.txt"):
        # get raw data
        if name == 'wmt14':
            dataset_dict = load_dataset("wmt14", "-".join(language_pair))
            train, val, test = (dataset_dict['train']['translation'], 
                                dataset_dict['validation']['translation'], 
                                dataset_dict['test']['translation'])
            raw_train_data = [tuple(sentence_pair.values()) for sentence_pair in train]
            raw_val_data = [tuple(sentence_pair.values()) for sentence_pair in val]
            raw_test_data = [tuple(sentence_pair.values()) for sentence_pair in test]
        elif name == 'm30k':
            dataset_dict = load_dataset("bentrevett/multi30k")
            train, val, test = (dataset_dict['train'], 
                                dataset_dict['validation'], 
                                dataset_dict['test'])
            raw_train_data = [tuple(sentence_pair.values()) for sentence_pair in train]
            raw_val_data = [tuple(sentence_pair.values()) for sentence_pair in val]
            raw_test_data = [tuple(sentence_pair.values()) for sentence_pair in test]
        elif name == 'txt':
            with open(dataset_path, 'r') as f:
                file_text = f.readlines()
                src_sentences = [sentence_pair.split("|")[0].strip() for sentence_pair in file_text]
                tgt_sentences = [sentence_pair.split("|")[1].strip() for sentence_pair in file_text]
                raw_data = list(zip(src_sentences, tgt_sentences))
                raw_train_data, raw_val_data, raw_test_data = DataProcessor.split_data(raw_data, 
                                                                                       split_ratio=(0.8, 0.1, 0.1), 
                                                                                       random_seed=random_seed)
        else: 
            raise ValueError(f"Received {name}, available options: 'wmt14', 'm30k', 'txt'")
        # save raw data
        raw_data = {
            'train': raw_train_data,
            'val': raw_val_data,
            'test': raw_test_data
        }
        return raw_data
