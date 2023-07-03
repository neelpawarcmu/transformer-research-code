import argparse
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from model.full_model import TransformerModel
from dataset_utils import load_tokenizers, load_vocab, create_dataloaders, Batch

class LabelSmoothing(nn.Module):
    """
    Implement label smoothing. TODO: improve this docstring
    """
    def __init__(self, vocab_size, pad_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.vocab_size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        mask = torch.nonzero(target.data == self.pad_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class SimpleLossCompute:
    """
    A simple loss compute and train function.
    """
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, y_preds, y_labels, batch_size):
        sloss = self.criterion(
                y_preds.contiguous().view(-1, y_preds.size(-1)), 
                y_labels.contiguous().view(-1)
            ) / batch_size
        return sloss.data * batch_size, sloss

def create_model(src_vocab_size: int,
                 tgt_vocab_size: int,
                 N: int = 6, 
                 d_model: int = 512,
                 d_ff: int = 2048,
                 h: int = 8, 
                 dropout_prob: float = 0.1):
    model = TransformerModel(src_vocab_size, tgt_vocab_size, N, d_model, 
                             d_ff, h, dropout_prob)
    return model

def create_config(vocab_src, vocab_tgt, args):
    config = {
        "pad_idx": vocab_tgt["<blank>"],
        "batch_size": 32,
        "N": 6,
        "d_model": 512,
        "d_ff": 2048,
        "h": 8,
        "dropout_prob": 0.1,
        "num_epochs": args.epochs,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "model_save_name": "artifacts/saved_models/multi30k_model",
        "src_vocab_size": len(vocab_src),
        "tgt_vocab_size": len(vocab_tgt),
    }
    return config

def get_learning_rate(step_num, d_model, factor, warmup):
    """
    Compute the learning rate from the equation (3) in section 5.3
    of the paper.
    """
    return factor * (
        d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup ** (-1.5))
    )

def train(train_dataloader, valid_dataloader, model, criterion, 
          optimizer, scheduler, config):
    

    for epoch in range(config["num_epochs"]):
        # create batched data loaders
        batched_train_dataloader = (Batch(b[0], b[1], config["pad_idx"]) 
                                    for b in train_dataloader)
        batched_valid_dataloader = (Batch(b[0], b[1], config["pad_idx"]) 
                                    for b in valid_dataloader)
        # initialize timer
        start = time.time()
        # training
        train_loss = run_train_epoch(batched_train_dataloader, model, criterion, optimizer, 
                                     scheduler, config["accum_iter"])
        # validation
        valid_loss = run_valid_epoch(batched_valid_dataloader, model, criterion)
        print(f"Epoch: {epoch} | "
              f"Average training loss: {train_loss / len(train_dataloader):.3f} | "
              f"Average validation loss: {valid_loss / len(valid_dataloader):.3f} | "
              f"Time taken: {1/60*(time.time() - start):.2f} min")
        print("="*80)

        # save model
        torch.save(model.state_dict(), f'{config["model_save_name"]}_epoch_{epoch}.pt')




def run_train_epoch(batched_train_dataloader, model, criterion, optimizer, scheduler, 
                    accum_iter):
    # put model in training mode
    model.train()
    # iterate over the training data and compute losses
    total_loss = 0
    for i, batch in enumerate(batched_train_dataloader):
        output_probabilities = model.forward(batch.src, batch.tgt, batch.decoder_attn_mask)
        loss, loss_node = criterion(output_probabilities, batch.tgt_y, batch.ntokens)
        loss_node.backward()
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss / batch.ntokens
        scheduler.step()
        
        # print metrics
        if i % 100 == 0:
            print(f"Step: {i} \t|\t Training loss: {loss/batch.ntokens:.3f} \t|\t"
                  f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    return total_loss

def run_valid_epoch(batched_valid_dataloader, model, criterion):
    # put model in evaluation mode
    model.eval()
    # iterate over the validation data and compute losses
    total_loss = 0
    for i, batch in enumerate(batched_valid_dataloader):
        output_probabilities = model.forward(batch.src, batch.tgt, batch.decoder_attn_mask)
        loss, loss_node = criterion(output_probabilities, batch.tgt_y, batch.ntokens)
        total_loss += loss / batch.ntokens
    
    return total_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load vocabulary
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    # create configuration for model and training
    config = create_config(vocab_src, vocab_tgt, args)
    
    # initialize model
    model = create_model(config["src_vocab_size"], config["tgt_vocab_size"])
    model = model.to(device)
    
    # load data
    train_dataloader, valid_dataloader = create_dataloaders(device, vocab_src, 
                                                            vocab_tgt, spacy_de, 
                                                            spacy_en, 
                                                            batch_size=config["batch_size"], 
                                                            max_padding=config["max_padding"])

    # create loss criterion, learning rate optimizer and scheduler
    label_smoothing = LabelSmoothing(config["tgt_vocab_size"], config["pad_idx"], 0.1)
    label_smoothing = label_smoothing.to(device)
    criterion = SimpleLossCompute(label_smoothing)
    # criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], 
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer = optimizer, 
                            lr_lambda = lambda step_num: get_learning_rate(
                                step_num+1, config["d_model"], factor=1, warmup=config["warmup"])
    )
    
    # train
    train(train_dataloader, valid_dataloader, model, criterion, optimizer,
          scheduler, config)