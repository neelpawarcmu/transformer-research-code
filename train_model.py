import argparse
import time
import json
import torch
from torch.optim.lr_scheduler import LambdaLR
from model.full_model import TransformerModel
from vocab.vocab_utils import build_tokenizers, build_vocabularies
from data.processors import DataProcessor
from data.runtime_loaders import load_datasets, load_dataloaders
from training.save_utils import SaveDirs
from training.loss_utils import SimpleLossCompute, LabelSmoothing
from training.loggers import Logger
from inference.utils import Translate

def create_model(src_vocab_size: int,
                 tgt_vocab_size: int,
                 N: int, 
                 d_model: int = 512,
                 d_ff: int = 2048,
                 h: int = 8, 
                 dropout_prob: float = 0.1):
    model = TransformerModel(src_vocab_size, tgt_vocab_size, N, d_model, 
                             d_ff, h, dropout_prob)
    return model

def create_config(args):
    config = {
        "language_pair": args.language_pair,
        "N": args.N,
        "batch_size": 64,
        "d_model": 512,
        "d_ff": 2048,
        "h": 8,
        "dropout_prob": 0.1,
        "num_epochs": args.epochs,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 20,
        "warmup": 3000,
        "model_dir": f"artifacts/saved_models",
    }
    # save config as a json file
    with open('training/config.json', 'w') as fp:
        json.dump(config, fp)
    return config

def get_learning_rate(step_num, d_model, warmup):
    """
    Compute the learning rate from the equation (3) in section 5.3
    of the paper.
    """
    learning_rate = d_model**-0.5 * min(step_num**-0.5, step_num*warmup**-1.5)
    return learning_rate

def train(train_dataloader, val_dataloader, model, criterion, 
          optimizer, scheduler, config):
    
    # initiate logger for saving metrics
    logger = Logger(N=config["N"])
    for epoch in range(1, config["num_epochs"]+1): # epochs are 1-indexed
        # initialize timer
        start = time.time()
        # training
        train_loss, train_bleu = run_train_epoch(train_dataloader, 
                                                 model, 
                                                 criterion, 
                                                 optimizer, 
                                                 scheduler, 
                                                 config["accum_iter"])
        # validation
        val_loss, val_bleu = run_val_epoch(val_dataloader, model, criterion)

        # Accumulate loss history, train loss should be the latest train loss  
        # since validation is done at end of entire training epoch
        logger.log('train_loss', train_loss)
        logger.log('val_loss', val_loss)

        # print losses
        print(f"Epoch: {epoch} | "
              f"Average training loss: {train_loss:.3f}, BLEU: {train_bleu:.3f} | "
              f"Average validation loss: {val_loss:.3f}, BLEU: {val_bleu:.3f} | "
              f"Time taken: {1/60*(time.time() - start):.2f} min")
        print("="*80)

        # save model
        torch.save(model.state_dict(),
                   f'{config["model_dir"]}/N{config["N"]}/epoch_{epoch:02d}.pt')

    # plot and save loss curves
    # logger.plot()

def run_train_epoch(train_dataloader, model, criterion, optimizer, 
                    scheduler, accum_iter):
    # put model in training mode
    model.train()
    # iterate over the training data and compute losses
    total_loss, total_bleu = 0, 0
    for i, batch in enumerate(train_dataloader):
        output_probabilities = model.forward(batch.src, 
                                             batch.tgt_shifted_right,
                                             batch.decoder_attn_mask)
        # compute loss and BLEU score
        loss = criterion(output_probabilities, batch.tgt_label, batch.ntokens)
        # TODO: passing global variable for 'translate' below should be resolved
        batch.predictions = torch.argmax(output_probabilities, dim=2)   
        bleu = translate.compute_batch_bleu(batch)
        # backpropagate and apply optimizer-based gradient descent 
        loss.backward()
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # step learning rate
        scheduler.step()
        # accumulate loss and BLEU score
        total_loss += loss.detach().cpu().numpy()
        total_bleu += bleu
        # print metrics
        if i % (len(train_dataloader)//8) == 0:
            print(f"Batch: {i}/{len(train_dataloader)} \t|\t"
                  f"Training loss: {loss.detach().cpu().numpy():.3f} \t|\t"
                  f"BLEU: {bleu:.3f} \t|\t"
                  f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    # average the metrics
    epoch_loss = total_loss / len(train_dataloader)
    epoch_bleu = total_bleu / len(train_dataloader)
    return epoch_loss, epoch_bleu

def run_val_epoch(val_dataloader, model, criterion):
    # put model in evaluation mode
    model.eval()
    # iterate over the validation data and compute losses
    total_loss, total_bleu = 0, 0
    for batch in val_dataloader:
        output_probabilities = model.forward(batch.src, batch.tgt_shifted_right, 
                                             batch.decoder_attn_mask)
        # compute loss and BLEU score
        loss = criterion(output_probabilities, batch.tgt_label, batch.ntokens)
        # TODO: passing global variable for 'translate' here should be resolved
        batch.predictions = torch.argmax(output_probabilities, dim=2)   
        bleu = translate.compute_batch_bleu(batch)
        # accumulate loss and BLEU score
        total_loss += loss.detach().cpu().numpy()
        total_bleu += bleu
    
    # average the metrics
    epoch_loss = total_loss / len(val_dataloader)
    epoch_bleu = total_bleu / len(val_dataloader)
    return epoch_loss, epoch_bleu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--language_pair", type=tuple, default=("de", "en"))
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create configuration for training
    config = create_config(args)

    # create if missing, directories required for saving artifacts
    SaveDirs.create_dirs(['saved_vocab', 'saved_data', 
                          f'saved_models/N{args.N}', f'loss_curves/N{args.N}'])

    # load tokenizers and vocabulary
    tokenizer_src, tokenizer_tgt = build_tokenizers(config["language_pair"])
    vocab_src, vocab_tgt = build_vocabularies(tokenizer_src, tokenizer_tgt,
                                              data = DataProcessor.get_raw_data(config["language_pair"]))

    # initialize model
    model = create_model(vocab_src.length, vocab_tgt.length, config["N"])
    model = model.to(device)
    
    # load data
    train_dataset, val_dataset, test_dataset = load_datasets(config["language_pair"],
                                                             tokenizer_src, 
                                                             tokenizer_tgt, 
                                                             vocab_src,
                                                             vocab_tgt,
                                                             config["max_padding"],
                                                             device=device,
                                                             random_seed=4)

    train_dataloader, val_dataloader, _ = load_dataloaders(train_dataset, 
                                                           val_dataset, 
                                                           test_dataset,
                                                           config["batch_size"],
                                                           shuffle=True)
    
    # create loss criterion, learning rate optimizer and scheduler
    label_smoothing = LabelSmoothing(vocab_tgt.length, vocab_tgt["<blank>"], 0.1)
    label_smoothing = label_smoothing.to(device)
    criterion = SimpleLossCompute(label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], 
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer = optimizer, 
                         lr_lambda = lambda step_num: get_learning_rate(
                         step_num+1, config["d_model"], 
                         warmup=config["warmup"]))
    
    # initialize translation utils TODO: check if this needs to go somewhere else / needs refactoring
    translate = Translate(vocab_src, vocab_tgt)

    # train
    train(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, config)