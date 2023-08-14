import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext.datasets as datasets
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.optim.lr_scheduler import LambdaLR
from model.full_model import TransformerModel
from old_dataset_utils import Batch
from data_utils.vocab_utils import build_tokenizers, build_vocabularies
from data_utils.dataset_utils import load_datasets
from path_utils import SaveDirs
from training_utils.loss_utils import SimpleLossCompute, LabelSmoothing

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

def train(train_dataloader, val_dataloader, model, criterion, 
          optimizer, scheduler, config):
    
    train_history, val_history = [], []
    for epoch in range(1, config["num_epochs"]+1): # epochs are 1-indexed
        # create batched data loaders
        batched_train_dataloader = (Batch(b[0], b[1], config["pad_idx"]) 
                                    for b in train_dataloader)
        batched_val_dataloader = (Batch(b[0], b[1], config["pad_idx"]) 
                                    for b in val_dataloader)
        # initialize timer
        start = time.time()
        # training
        total_train_loss, epoch_latest_train_loss = run_train_epoch(
            batched_train_dataloader, 
            model, criterion, optimizer, scheduler, 
            config["accum_iter"]
        )
        epoch_avg_train_loss = total_train_loss / len(train_dataloader)
        # validation
        total_val_loss = run_val_epoch(batched_val_dataloader, model, 
                                           criterion)
        epoch_avg_val_loss = total_val_loss / len(val_dataloader)

        # Accumulate loss history, train loss should be the latest train loss  
        # since validation is done at end of entire training epoch
        train_history.append(epoch_avg_train_loss.cpu().numpy())
        val_history.append(epoch_avg_val_loss.cpu().numpy())

        # print losses
        print(f"Epoch: {epoch} | "
              f"Latest training loss: {epoch_latest_train_loss:.3f} | "
              f"Average training loss: {epoch_avg_train_loss:.3f} | \n"
              f"Average validation loss: {epoch_avg_val_loss:.3f} | "
              f"Time taken: {1/60*(time.time() - start):.2f} min")
        print("="*80)

        # save model
        torch.save(model.state_dict(), 
                   f'{config["model_save_name"]}_epoch_{epoch}.pt')

    # plot and save loss curves
    plot_losses(train_history, val_history)

def run_train_epoch(batched_train_dataloader, model, criterion, optimizer, 
                    scheduler, accum_iter):
    # put model in training mode
    model.train()
    # iterate over the training data and compute losses
    total_loss, latest_loss = 0, 0
    for i, batch in enumerate(batched_train_dataloader):
        # import pdb; pdb.set_trace()
        output_probabilities = model.forward(batch.src, 
                                             batch.tgt_shifted_right,
                                             batch.decoder_attn_mask)
        loss = criterion(output_probabilities, batch.tgt_label, batch.ntokens)
        loss.backward()
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        latest_loss = loss.data
        total_loss += latest_loss
        scheduler.step()
        
        # print metrics
        if i % 100 == 0:
            print(f"Batch: {i} \t|\t Training loss: {latest_loss:.3f} \t|\t"
                  f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    return total_loss, latest_loss

def run_val_epoch(batched_val_dataloader, model, criterion):
    # put model in evaluation mode
    model.eval()
    # iterate over the validation data and compute losses
    total_loss = 0
    for i, batch in enumerate(batched_val_dataloader):
        output_probabilities = model.forward(batch.src, batch.tgt_shifted_right, 
                                             batch.decoder_attn_mask)
        loss = criterion(output_probabilities, 
                              batch.tgt_label, 
                              batch.ntokens)
        loss.backward()
        total_loss += loss.data
    
    return total_loss

def plot_losses(train_history, val_history):
    plt.figure(dpi=300)
    plt.plot(train_history, label="training loss")
    plt.plot(val_history, label="validation loss")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.grid(visible=True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("artifacts/loss_curves/loss_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create required directories for saving artifacts
    SaveDirs.create_train_dirs()

    # load tokenizers and vocabulary
    tokenizer_src, tokenizer_tgt = build_tokenizers()
    vocab_src, vocab_tgt = build_vocabularies(tokenizer_src, tokenizer_tgt)

    # create configuration for model and training
    config = create_config(vocab_src.vocab, vocab_tgt.vocab, args)

    # initialize model
    model = create_model(config["src_vocab_size"], config["tgt_vocab_size"])
    model = model.to(device)
    
    # load data
    train_dataset, val_dataset, test_dataset = load_datasets(
        tokenizer_src, 
        tokenizer_tgt, 
        vocab_src.vocab,
        vocab_tgt.vocab, 
        config,
        device,
        preprocess=True
    )

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=config["batch_size"], 
                                  shuffle=True,
                                  collate_fn=train_dataset.collate_fn)
    
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=config["batch_size"], 
                                shuffle=True,
                                collate_fn=val_dataset.collate_fn)
    
    # create loss criterion, learning rate optimizer and scheduler
    label_smoothing = LabelSmoothing(config["tgt_vocab_size"], config["pad_idx"], 0.1)
    label_smoothing = label_smoothing.to(device)
    criterion = SimpleLossCompute(label_smoothing)
    
    # criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], 
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer = optimizer, 
                            lr_lambda = lambda step_num: get_learning_rate(
                                step_num+1, config["d_model"], factor=1, 
                                warmup=config["warmup"])
    )
    
    # train
    train(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, config)