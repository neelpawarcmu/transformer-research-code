import torch
from torch.utils.data import DataLoader
from data.processors import DataProcessor

# "Batching"
class Batch:
    """
    Object for holding a batch of data with mask during training.
    """
    def __init__(self, src, tgt=None, pad_idx=2):  # 2 = <blank>
        self.src = src
        if tgt is not None:
            self.tgt_shifted_right = tgt[:, :-1] # everything except last pad token
            self.tgt_label = tgt[:, 1:] # everything except beginning of sentence token
            self.decoder_attn_mask = self.make_decoder_attn_mask(
                self.tgt_shifted_right, pad_idx
            )
            self.ntokens = (self.tgt_label != pad_idx).sum()

    @staticmethod
    def make_decoder_attn_mask(tgt, pad_idx):
        """
        Create a mask to hide padding and future words.
        TODO: explain multi mask creation for entire batch
        """
        pad_mask = (tgt != pad_idx).unsqueeze(-2)
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
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.length = dataset.shape[0]
        self.device = device
    
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
        return Batch(batch_src, batch_tgt)
    
        
def load_datasets(language_pair, tokenizer_src, tokenizer_tgt, vocab_src, 
                  vocab_tgt, max_padding, device, random_seed=None):
    '''
    A utility function that sources the preprocessed data, calls a split on 
    it, generates runtime dataset splits for training, validation and testing.
    '''
    data_processor = DataProcessor(tokenizer_src,
                                   tokenizer_tgt,
                                   vocab_src,
                                   vocab_tgt,
                                   max_padding,
                                   language_pair)
    preprocd_data = data_processor.get_preprocessed_data()
    
    (preprocd_train_data, 
     preprocd_val_data, 
     preprocd_test_data) = data_processor.get_data_splits(
        preprocd_data, random_seed=random_seed
    )
    
    train_dataset = RuntimeDataset(preprocd_train_data, device)
    val_dataset = RuntimeDataset(preprocd_val_data, device)
    test_dataset = RuntimeDataset(preprocd_test_data, device)

    return train_dataset, val_dataset, test_dataset

    
def load_dataloaders(train_dataset, val_dataset, test_dataset, 
                     batch_size, shuffle=True):
    '''
    A utility function that takes runtime dataset splits and creates 
    corresponding train, validation and test dataloaders that consume the 
    dataset splits and batch them at runtime for the model.
    '''
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle,
                                  collate_fn=train_dataset.collate_fn)
    
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle,
                                collate_fn=val_dataset.collate_fn)
    
    test_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle,
                                collate_fn=val_dataset.collate_fn)
    return train_dataloader, val_dataloader, test_dataloader