import os
import torch
from torch.utils.data import DataLoader
from data.download import DataDownloader
from data.processors import DataProcessor

# "Batching"
class Batch:
    """
    Object for holding a batch of data with mask during training.
    """
    def __init__(self, pad_idx_tgt, src, tgt=None):
        self.src = src
        if tgt is not None:
            self.tgt_shifted_right = tgt[:, :-1] # everything except last pad token
            self.tgt_label = tgt[:, 1:] # everything except beginning of sentence token
            self.decoder_attn_mask = self.make_decoder_attn_mask(
                self.tgt_shifted_right, pad_idx_tgt
            )
            self.ntokens = (self.tgt_label != pad_idx_tgt).sum()

    @staticmethod
    def make_decoder_attn_mask(tgt, pad_idx_tgt):
        """
        Create a mask to hide padding and future words.
        TODO: explain multi mask creation for entire batch
        """
        pad_mask = (tgt != pad_idx_tgt).unsqueeze(-2)
        pad_mask_T = pad_mask.transpose(1,2)
        subseq_tokens_mask = Batch.get_subseq_tokens_mask(tgt)
        decoder_attn_mask = pad_mask & subseq_tokens_mask & pad_mask_T
        # TODO remove
        # visualize_attn_mask(decoder_attn_mask)
        return decoder_attn_mask
    
    @staticmethod
    def get_subseq_tokens_mask(tgt):
        """
        Generate an upper triangular matrix to mask out subsequent positions
        """
        mask_shape = (1, tgt.size(-1), tgt.size(-1))
        upper_tri_matrix = torch.triu(torch.ones(mask_shape), diagonal=1)
        subseq_tokens_mask = (upper_tri_matrix == 0).type_as(tgt)
        return subseq_tokens_mask
    
class RuntimeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device, pad_idx_tgt):
        self.dataset = dataset
        self.length = dataset.shape[0]
        self.device = device
        self.pad_idx_tgt = pad_idx_tgt
    
    def __getitem__(self, i):
        '''
        Return a tensor corresponding to a single pair of sentences
        '''
        tok_sentence_pair = self.dataset[i]
        return tok_sentence_pair
    
    def __len__(self):
        '''
        Return length of entire dataset
        '''
        return self.length
    
    def collate_fn(self, raw_batch):
        '''
        Collate a batch from N preprocessed data samples, where N is 
        the batch size specified in the dataloader.
        '''
        batch_tensor = torch.stack(raw_batch)
        batch_src = batch_tensor[:,0,:].to(self.device)
        batch_tgt = batch_tensor[:,1,:].to(self.device)
        return Batch(self.pad_idx_tgt, batch_src, batch_tgt)
    
        
def load_datasets(name, language_pair, tokenizer_src, tokenizer_tgt,
                  max_padding, device, cache, random_seed=None, 
                  dataset_size=5000000):
    '''
    A utility function that sources the preprocessed data, calls a split on 
    it, generates runtime dataset splits for training, validation and testing.
    '''
    print(f'loading dataset {name}')
    data_processor = DataProcessor(tokenizer_src,
                                   tokenizer_tgt,
                                   max_padding,
                                   language_pair)
    preprocd_data = DataDownloader.get_data(name=name, 
                                            language_pair=language_pair, 
                                            cache=cache, 
                                            preprocess=True, 
                                            preprocessor=data_processor,
                                            dataset_size=dataset_size)
    
    (preprocd_train_data, 
     preprocd_val_data, 
     preprocd_test_data) = data_processor.get_data_splits(
        preprocd_data, random_seed=random_seed
    )
    
    train_dataset = RuntimeDataset(preprocd_train_data, device, tokenizer_tgt.pad_token_id)
    val_dataset = RuntimeDataset(preprocd_val_data, device, tokenizer_tgt.pad_token_id)
    test_dataset = RuntimeDataset(preprocd_test_data, device, tokenizer_tgt.pad_token_id)

    print(f"Number of sentence pairs: \n"
          f"Training: {train_dataset.length}\t"
          f"Validation: {val_dataset.length}\t"
          f"Test: {test_dataset.length}\t")

    return train_dataset, val_dataset, test_dataset

    
def load_dataloaders(train_dataset, val_dataset, test_dataset, 
                     batch_size, shuffle=True, num_workers=1):
    '''
    A utility function that takes runtime dataset splits and creates 
    corresponding train, validation and test dataloaders that consume the 
    dataset splits and batch them at runtime for the model.
    '''
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle,
                                  collate_fn=train_dataset.collate_fn,
                                #   num_workers=5,
                                #   persistent_workers=True,
                                #   pin_memory=True,
                                #   prefetch_factor=3,
                                )
    
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle,
                                collate_fn=val_dataset.collate_fn,
                                # num_workers=5,
                                # persistent_workers=True,
                                # pin_memory=True,
                                # prefetch_factor=3,
                                )
    
    test_dataloader = DataLoader(dataset=test_dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle,
                                collate_fn=val_dataset.collate_fn)
    return train_dataloader, val_dataloader, test_dataloader