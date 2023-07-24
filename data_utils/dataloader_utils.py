import torch
# from os.path import exists
from torchtext.data.functional import to_map_style_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from tqdm import tqdm

class CustomDataset:
    def __init__(self, 
                 raw_data, 
                 tokenizer_src, 
                 tokenizer_tgt, 
                 vocab_src, 
                 vocab_tgt,
                 max_padding,
                 device):
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.device = device
        self.max_padding = max_padding
        self.dataset = self.preprocess_dataset(raw_data)
    
    def preprocess_sentence(self, sentence, tokenizer, vocab):
        """
        Performs the following operations:
        - tokenize sentence
        - add beginning of sentence and end of sentence tokens
        - pad end of sentence up to max_padding
        """
        bos_id, eos_id = 0, 1 # beginning, end of sentence: <s>, </s>
        pad_id = 2 # id for pad tokens
        # tokenize sentence into individual tokens
        tokens = tokenizer.tokenize(sentence)
        # map tokens to ids
        token_ids = vocab(tokens)
        # convert to tensor for compatibility with padding function
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.int64)
        # pad token ids
        pad_size = (0, self.max_padding - len(token_ids_tensor))
        padded_token_ids = pad(token_ids_tensor, pad_size, value=pad_id)
        return padded_token_ids

    def preprocess_dataset(self, raw_data):
        """
        Preprocess dataset, return preprocessed tensor
        """
        data_map = to_map_style_dataset(raw_data)
        # tokenize the dataset
        preprocessed_dataset = torch.zeros(
            size=(len(data_map), len(data_map[0]), self.max_padding),
            dtype=torch.int64,
        )
        
        for i, (src, tgt) in enumerate(tqdm(data_map, desc="Preprocessing dataset")):
            preprocessed_src = self.preprocess_sentence(src, 
                                                        self.tokenizer_src,
                                                        self.vocab_src)
            preprocessed_tgt = self.preprocess_sentence(tgt, 
                                                        self.tokenizer_tgt,
                                                        self.vocab_tgt)
            preprocessed_dataset[i][0] = preprocessed_src
            preprocessed_dataset[i][1] = preprocessed_tgt
        
        return preprocessed_dataset            

    # def collate_fn(self, batch):

    # def load_dataset()