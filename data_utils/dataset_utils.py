import torch
from torch.nn.functional import pad
import torchtext.datasets as datasets
from sklearn.model_selection import train_test_split
from torchtext.data.functional import to_map_style_dataset
from tqdm import tqdm

class DatasetPreprocessor:
    def __init__(self,
                 tokenizer_src, 
                 tokenizer_tgt, 
                 vocab_src, 
                 vocab_tgt,
                 max_padding):
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_padding = max_padding

    def tensor_to_sentence(self, sentence, tokenizer, vocab):
        '''
        Reverse operations to those described under 
        :meth:`<dataset_utils.DatasetPreprocessor.sentence_to_tensor>`
        '''
        # get ids for beginning and end of sentence, and padding markers
        bos_id, eos_id, pad_id = 0, 1, 2
        raise NotImplementedError

    def sentence_to_tensor(self, sentence, tokenizer, vocab):
        '''
        Performs the following operations:
        - tokenize sentence
        - add beginning of sentence and end of sentence markers
        - pad end of sentence up to max_padding
        '''
        # get ids for beginning and end of sentence, and padding markers
        bos_id, eos_id, pad_id = 0, 1, 2
        # tokenize sentence into individual tokens
        tokens = tokenizer.tokenize(sentence)
        # map tokens to ids and add beginning, end of sentence markers
        token_ids = [bos_id] + vocab(tokens) + [eos_id]
        # convert to tensor for compatibility with padding function
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.int64)
        # pad token ids
        pad_size = (0, self.max_padding - len(token_ids_tensor))
        padded_token_ids = pad(token_ids_tensor, pad_size, value=pad_id)
        return padded_token_ids

    def preprocess_dataset(self, raw_dataset):
        '''
        Preprocess dataset sentence by sentence, as detailed in 
        :meth:`<dataset_utils.DatasetPreprocessor.sentence_to_tensor>`
        '''
        data_map = to_map_style_dataset(raw_dataset) # map allows access to length of dataset
        # initialize an empty preprocessed dataset
        preprocessed_dataset = torch.zeros(
            size=(len(data_map), len(data_map[0]), self.max_padding),
            dtype=torch.int64,
        )
        # populate the preprocessed dataset, sentence pair by sentence pair
        for i, (src, tgt) in enumerate(tqdm(data_map, desc="Preprocessing dataset")):
            preprocessed_src = self.sentence_to_tensor(src, 
                                                        self.tokenizer_src,
                                                        self.vocab_src)
            preprocessed_tgt = self.sentence_to_tensor(tgt, 
                                                        self.tokenizer_tgt,
                                                        self.vocab_tgt)
            preprocessed_dataset[i][0] = preprocessed_src
            preprocessed_dataset[i][1] = preprocessed_tgt
        
        return preprocessed_dataset
    
    def get_preprocessed_dataset(self):
        raw_dataset = self.get_raw_dataset()
        preprocd_dataset = self.preprocess_dataset(raw_dataset)
        return preprocd_dataset
        
    @staticmethod
    def get_raw_dataset():
        train_iter, valid_iter, test_iter = datasets.Multi30k(language_pair=("de", "en"))
        raw_dataset = train_iter + valid_iter + test_iter
        return raw_dataset
    
    @staticmethod
    def get_dataset_splits(dataset, split_ratio=(0.8, 0.1, 0.1)):
        '''
        Splits a given dataset into train, validation and test sets as
        determined by the specified split ratio
        '''
        train_size, val_size, test_size = split_ratio
        train_dataset, val_and_test_dataset = train_test_split(
            dataset,
            train_size=train_size
        )
        val_dataset, test_dataset = train_test_split(
            val_and_test_dataset,
            train_size=val_size/(val_size+test_size)
        )
        return train_dataset, val_dataset, test_dataset
    
class RuntimeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.length = dataset.shape[0]
        self.device = device
    
    def __getitem__(self, i):
        '''
        Return a tensor corresponding to a single pair of sentences
        '''
        sentence_pair = self.dataset[i]
        return sentence_pair
    
    def __len__(self):
        return self.length
    
    def collate_fn(self, raw_batch):
        batch_tensor = torch.stack(raw_batch)
        batch_src = batch_tensor[:,0,:].to(self.device)
        batch_tgt = batch_tensor[:,1,:].to(self.device)
        return batch_src, batch_tgt
    
class RuntimeDataLoader(torch.utils.data.DataLoader):
    def __init__(self, 
                 dataset, 
                 batch_size,
                 shuffle,
                 collate_fn):
        super(RuntimeDataLoader, self).init(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            collate_fn=collate_fn)
        
def load_datasets(tokenizer_src,
                  tokenizer_tgt,
                  vocab_src,
                  vocab_tgt,
                  config,
                  device,
                  preprocess=True):
    
    dataset_preprocessor = DatasetPreprocessor(tokenizer_src,
                                               tokenizer_tgt,
                                               vocab_src,
                                               vocab_tgt,
                                               config["max_padding"])
    preprocd_dataset = dataset_preprocessor.get_preprocessed_dataset()
    
    (preprocd_train_dataset, 
     preprocd_val_dataset, 
     preprocd_test_dataset) = dataset_preprocessor.get_dataset_splits(preprocd_dataset)
    
    train_dataset = RuntimeDataset(preprocd_train_dataset, device)
    val_dataset = RuntimeDataset(preprocd_val_dataset, device)
    test_dataset = RuntimeDataset(preprocd_test_dataset, device)

    return train_dataset, val_dataset, test_dataset

    
def load_dataloaders():
    pass