import os
import time
import json
import torch
import torch.nn as nn
import argparse
from torch.optim.lr_scheduler import LambdaLR

from vocab.vocab_utils import build_tokenizers
from vocab.bert_tokenizer_utils import build_tokenizers
from data.runtime_loaders import load_datasets, load_dataloaders

from training.logging import DirectoryCreator, TrainingLogger
from training.loss import LabelSmoothing, SimpleLossCompute
from training.utils import get_learning_rate
from inference.utils import BleuUtils
from model.full_model import TransformerModel
from model.utils import count_params
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def create_model(config):
    model =  TransformerModel(config["src_vocab_size"], 
                              config["tgt_vocab_size"], 
                              config["N"], 
                              config["d_model"], 
                              config["d_ff"], 
                              config["h"], 
                              config["dropout_prob"])
    for layer in [model, 
                  model.input_embedding_layer, 
                  model.output_embedding_layer, 
                  model.input_positional_enc_layer, 
                  model.output_positional_enc_layer,
                  model.encoder_stack, 
                  model.decoder_stack, 
                  model.linear_and_softmax_layers]:
        count_params(layer)
    model = nn.DataParallel(model) # enable running on multiple GPUs
    return model

def create_config(args, src_vocab_size, tgt_vocab_size):
    config = {
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "dataset_name": args.dataset_name,
        "language_pair": tuple(args.language_pair),
        "N": args.N,
        "batch_size": args.batch_size,
        "d_model": 512,
        "d_ff": 2048,
        "h": 8,
        "dropout_prob": 0.1,
        "num_epochs": args.epochs,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": args.max_padding,
        "warmup": 3000,
        "model_dir": f"artifacts/saved_models",
        "dataset_size": args.dataset_size,
        "experiment_name": args.experiment_name,
        "random_seed": args.random_seed,
    }
    # save config as a json file
    with open('artifacts/training_config.json', 'w') as fp:
        json.dump(config, fp)
    return config

def train(train_dataloader, val_dataloader, model, criterion, 
          optimizer, scheduler, config):
    
    # initiate logger for saving metrics
    logger = TrainingLogger(config)
    # start training
    for epoch in range(1, config["num_epochs"]+1): # epochs are 1-indexed
        print(f"Epoch: {epoch}")
        # initialize timer
        start = time.time()
        # training
        train_loss, train_bleu = run_train_epoch(train_dataloader, 
                                                 val_dataloader,
                                                 model, 
                                                 criterion, 
                                                 optimizer, 
                                                 scheduler, 
                                                 config["accum_iter"],
                                                 logger)
        # validation
        val_loss, val_bleu = run_val_epoch(val_dataloader, model, criterion)

        # Accumulate loss history, train loss should be the latest train loss  
        # since validation is done at end of entire training epoch
        # logger.log('train_loss', train_loss)
        # logger.log('train_bleu', train_bleu)
        logger.log('val_loss', val_loss, epoch)
        logger.log('val_bleu', val_bleu, epoch)

        # print losses
        print(f"Epoch: {epoch} | "
              f"Training: Loss: {train_loss:.3f}, BLEU: {train_bleu:.3f} | "
              f"Validation: Loss: {val_loss:.3f}, BLEU: {val_bleu:.3f} | "
              f"Time taken: {1/60*(time.time() - start):.2f} min")
        print("="*100)

        # save model
        # torch.save(model.state_dict(),
        #    f'{config["model_dir"]}/N{config["N"]}/epoch_{epoch:02d}.pt')

        # plot and save loss curves
        # log loss v/s weight updates
        logger.saveplot(epoch, 
                        metric_names=['train_loss', 'val_loss'], 
                        title='Loss', 
                        title_dict={k:config[k] for k in ['batch_size', 'N', 'dataset_size']}, 
                        plot_type='loss', 
                        xlabel='Weight Update',
                        )
        logger.saveplot(epoch,
                        metric_names=['train_bleu', 'val_bleu'], 
                        title='BLEU', 
                        title_dict={k:config[k] for k in ['batch_size', 'N', 'dataset_size']}, 
                        plot_type='bleu', 
                        xlabel='Weight Update',
                        )
    print("Training complete")

def run_train_epoch(train_dataloader, val_dataloader, model, criterion, 
                    optimizer, scheduler, accum_iter, logger):
    # put model in training mode
    model.train()
    # iterate over the training data and compute losses
    total_loss, total_bleu = 0, 0
    print_frequency = max(len(train_dataloader) // 8, 1) # for printing progress
    for i, batch in enumerate(train_dataloader):
        output_logprobabilities = model.forward(batch.src, 
                                                batch.tgt_shifted_right,
                                                batch.decoder_attn_mask)
        # compute loss and BLEU score
        loss = criterion(output_logprobabilities, batch.tgt_label, batch.ntokens)
        predictions = torch.argmax(output_logprobabilities, dim=2)   
        # TODO: passing global variable for 'bleu_utils' below should be resolved
        bleu = bleu_utils.compute_batch_bleu(predictions, batch.tgt_label)
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
        
        # log train, val loss and bleu after every weight update
        # val_loss, val_bleu = run_val_epoch(val_dataloader, model, criterion)
        logger.log("train_loss", loss.item(), i)
        logger.log("train_bleu", bleu, i)
        # logger.log("val_loss", None, i)
        # logger.log("val_bleu", None, i)
        
        # print metrics
        if i % print_frequency == 0:
            print(f"Batch: {i+1}/{len(train_dataloader)} \t|\t"
                  f"Training loss: {loss.detach().cpu().numpy():.3f} \t|\t"
                  f"BLEU: {bleu:.3f} \t|\t"
                  f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        del batch
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
        output_logprobabilities = model.forward(batch.src, batch.tgt_shifted_right, 
                                             batch.decoder_attn_mask)
        # compute loss and BLEU score
        loss = criterion(output_logprobabilities, batch.tgt_label, batch.ntokens)
        # TODO: passing global variable for 'bleu_utils' here should be resolved
        predictions = torch.argmax(output_logprobabilities, dim=2)   
        bleu = bleu_utils.compute_batch_bleu(predictions, batch.tgt_label)
        # accumulate loss and BLEU score
        total_loss += loss.detach().cpu().numpy()
        total_bleu += bleu
    
    # average the metrics
    epoch_loss = total_loss / len(val_dataloader)
    epoch_bleu = total_bleu / len(val_dataloader)
    return epoch_loss, epoch_bleu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--language_pair", type=tuple, default=("de", "en"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_padding", type=int, default=20)
    parser.add_argument("--dataset_name", type=str, choices=["wmt14", "m30k"], default="wmt14")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--dataset_size", type=int, default=5000000)
    parser.add_argument("--random_seed", type=int, default=40)
    parser.add_argument("--experiment_name", type=str, default="default experiment")
    # parser.add_argument("--run_name", type=str, default="default run")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if missing, create directories required for saving artifacts
    DirectoryCreator.create_dirs(['saved_tokenizers', 
                                  'saved_data',
                                  f'saved_models/N{args.N}', 
                                  f'loss_curves/N{args.N}'])
    # load tokenizers and vocabulary
    tokenizer_src, tokenizer_tgt = build_tokenizers(args.language_pair)
    config = create_config(args, len(tokenizer_src.vocab), len(tokenizer_tgt.vocab))

    # initialize model
    model = create_model(config).to(device) # 
    
    # load data
    train_dataset, val_dataset, test_dataset = load_datasets(config["dataset_name"],
                                                             config["language_pair"],
                                                             tokenizer_src, 
                                                             tokenizer_tgt,
                                                             config["max_padding"],
                                                             device=device,
                                                             cache=args.cache,
                                                             random_seed=args.random_seed,
                                                             dataset_size=args.dataset_size)

    train_dataloader, val_dataloader, test_dataloader = load_dataloaders(train_dataset, 
                                                                         val_dataset, 
                                                                         test_dataset,
                                                                         config["batch_size"],
                                                                         shuffle=True,
                                                                         num_workers=os.cpu_count())
    # create loss criterion, learning rate optimizer and scheduler
    label_smoothing = LabelSmoothing(len(tokenizer_tgt.vocab), 
                                     tokenizer_tgt.pad_token_id, 0.1)
    label_smoothing = label_smoothing.to(device)
    criterion = SimpleLossCompute(label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], 
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer = optimizer, 
                         lr_lambda = lambda step_num: get_learning_rate(
                         step_num+1, config["d_model"], 
                         warmup=config["warmup"]))
    
    # initialize translation utils TODO: check if this needs to go somewhere else / needs refactoring
    bleu_utils = BleuUtils(tokenizer_src, tokenizer_tgt)
    # train
    train(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, config)