import json
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vocab.vocab_utils import build_tokenizers, build_vocabularies
from data.processors import DataProcessor
from data.runtime_loaders import load_datasets
from data.processors import SentenceProcessor
from model.full_model import TransformerModel
from training.save_utils import SaveDirs
from inference.utils import greedy_decode, Translate

def check_outputs(valid_dataloader, model, vocab_src, vocab_tgt, 
                  num_examples, config):
    results = [()] * num_examples
    print_text = ""
    print_text += f"Transformer layers: {config['N']} | Epoch: {config['epoch']}\n\n"
    model.eval() 

    for idx, batch in list(enumerate(valid_dataloader))[:num_examples]:
        model_pred_tokens = greedy_decode(model, batch, vocab_tgt)
        src_sentence = SentenceProcessor.tokens_to_sentence(batch.src[0], vocab_src)
        tgt_sentence = SentenceProcessor.tokens_to_sentence(batch.tgt_shifted_right[0], vocab_tgt)
        model_pred_sentence = SentenceProcessor.tokens_to_sentence(model_pred_tokens[0], vocab_tgt)
        print_text += (
            f"Example {idx+1} ========\n" + 
            f"Source Text (Input): {src_sentence}\n" +
            f"Target Text (Ground Truth): {tgt_sentence}\n" +
            f"Model Output: {model_pred_sentence}\n\n"
        )
        results[idx] = (src_sentence, tgt_sentence, model_pred_sentence)
    return results, print_text

def run_model_example(vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt, 
                      config):

    print("Preparing Data ...")
    _, val_dataset, _ = load_datasets(tokenizer_src, tokenizer_tgt, vocab_src,
                                      vocab_tgt, max_padding=50,
                                      device=torch.device("cpu"),
                                      random_seed=4)
    # TODO: remove redundancy and use dataloader creation util functions
    valid_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  collate_fn=val_dataset.collate_fn)
    
    print("Loading Trained Model ...")
    model = TransformerModel(vocab_src.length, vocab_tgt.length, N=config["N"])
    # load saved model weights
    model_path = f"{config['model_dir']}/N{config['N']}/epoch_{config['epoch']:02d}.pt"
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs ...")
    results, print_text = check_outputs(valid_dataloader, model, vocab_src,
                                        vocab_tgt, config["num_examples"], 
                                        config)
    bleu_score = Translate.compute_bleu(results)
    print_text += (f'BLEU Score: {bleu_score:.4f}')
    print(print_text)

    print("Saving Translations ...")
    save_translations(print_text, 
                      save_path=f"artifacts/generated_translations/"+
                      f"N{config['N']}/epoch_{config['epoch']:02d}.png")

def save_translations(print_text, save_path):
    """
    Save translation text as an image
    """
    plt.figure()
    plt.text(0, 1, print_text)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1
                        ) # 1-indexed epoch number of saved model
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--N", type=int, default=None)
    args = parser.parse_args()

    # load training config file
    with open('training/config.json', 'r') as fp:
        config = json.load(fp)
        config["epoch"] = args.epoch
        config["num_examples"] = args.num_examples
        if args.N: config['N'] = args.N

    # create directories required for saving artifacts
    SaveDirs.add_dir(f"generated_translations/N{config['N']}", include_base_path=True)

    # load vocabulary
    tokenizer_src, tokenizer_tgt = build_tokenizers("de", "en")
    vocab_src, vocab_tgt = build_vocabularies(tokenizer_src, tokenizer_tgt,
                                              DataProcessor.get_raw_data(config["language_pair"]))

    run_model_example(vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt,
                      config)