import json
import argparse
import torch
from tqdm import tqdm
from tokenization.tokenizers import build_tokenizers
from data.dataloading import load_datasets, load_dataloaders
from training.logging import TranslationLogger
from inference.utils import greedy_decode, BleuUtils

class Translator:
    def __init__(self, args, config_path):
        '''
        Initializes the Translator class by creating required directories 
        and loading the runtime configs
        '''
        # load model and training configurations saved from training run
        self.load_config(config_path, args)
    
    def load_config(self, filepath, args):
        '''
        Load a saved configuration json file as a dictionary to be used 
        for model loading and translation.
        '''
        with open(filepath, 'r') as fp:
            config = json.load(fp)
        # add information from args
        for k, v in args.__dict__.items():
            config[k] = v
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

    def prepare_tokenizers(self): 
        '''
        Load tokenizers and vocabularies
        '''
        tokenizer_src, tokenizer_tgt = build_tokenizers(self.config)
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

    def prepare_model(self):
        if HARVARD:
            from harvard import load_model
            save_path = f"{self.config['model_dir']}/N{self.config['N']}/harvard/{self.config['dataset_name']}/dataset_size_{self.config['dataset_size']}/epoch_{self.config['epoch']:02d}.pt"
            self.model = load_model(self.config, save_path)
        else:
            save_path = f"{self.config['model_dir']}/N{self.config['N']}/{self.config['dataset_name']}/dataset_size_{self.config['dataset_size']}/epoch_{self.config['epoch']:02d}.pt"
            self.model = torch.load(save_path)
        
        if isinstance(self.model, torch.nn.DataParallel):
            print("Unwrapping DataParallel model")
            self.model = self.model.module
        
        self.model.to(self.config["device"])
        # print(f"Loading model from {save_path}")

    def prepare_dataloader(self, split):
        train_dataset, val_dataset, test_dataset = load_datasets(
            self.config["dataset_name"],
            self.config["language_pair"], 
            self.tokenizer_src, 
            self.tokenizer_tgt, 
            self.config["max_padding"],
            device=self.config["device"],
            cache=True,
            random_seed=self.config["random_seed"],
            dataset_size=self.config["dataset_size"]
        )
        train_dataloader, val_dataloader, test_dataloader = load_dataloaders(
            train_dataset, 
            val_dataset, 
            test_dataset,
            self.config["batch_size"],
            shuffle=False
        )
        # select dataloader to use for translation
        self.dataloader = eval(f'{split}_dataloader')
        self.dataloader = list(self.dataloader)[:self.config["batch_limit"]]

    def translate_dataset(self):
        avg_bleu = 0
        pbar = tqdm(self.dataloader)
        with torch.inference_mode() and torch.autocast(device_type=self.config["device"].type, dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(pbar):
                pbar.set_description(f"Translating batch {batch_idx + 1} / {len(self.dataloader)}")
                bleu = self.translate_batch(self.model, batch, self.tokenizer_tgt, logger)
                avg_bleu += bleu
        avg_bleu /= len(self.dataloader)
        return avg_bleu
    
    def translate_batch(self, model, batch, tokenizer_tgt, logger=None):
        predictions = greedy_decode(model, batch, tokenizer_tgt, HARVARD)
        src_sentences = self.tokenizer_src.batch_decode(batch.src, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        tgt_sentences = self.tokenizer_tgt.batch_decode(batch.tgt_label, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predicted_sentences = self.tokenizer_tgt.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        bleu = BleuUtils.compute_batch_bleu(predicted_sentences, tgt_sentences)
        if logger:
            logger.log_sentence_batch(src_sentences, tgt_sentences, predicted_sentences, bleu)
        return bleu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1
                        ) # 1-indexed epoch number of saved model
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--split", type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument("--dataset_name", type=str, default='wmt14')
    parser.add_argument("--dataset_size", type=int, default=5000000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--batch_limit", type=int, default=100)
    parser.add_argument("--HARVARD", action='store_true')
    parser.add_argument("--tokenizer_type", type=str, default='bert', choices=['bert', 'spacy'])
    parser.add_argument("--random_seed", type=int, default=40)
    parser.add_argument("--max_padding", type=int, default=20)
    args = parser.parse_args()
    HARVARD = args.HARVARD
    # initialize Translator class
    translator = Translator(args, 'artifacts/training_config.json')
    # initialize logger
    logger = TranslationLogger(translator.config)
    # prepare fundamental blocks of translation pipeline
    translator.prepare_tokenizers()
    translator.prepare_dataloader(split=args.split)
    translator.prepare_model()
    # run translations  
    bleu_score = translator.translate_dataset()
    # save translation results
    logger.save_as_txt('artifacts/generated_translations/', 
                        title='Transformer translations',
                        title_dict={k:translator.config[k] for k in 
                                    ['N', 'epoch', 'num_examples', 
                                     'dataset_size', 'dataset_name']})
    # # print and save translation results
    # logger.print_and_save('artifacts/generated_translations/', 
    #                       title='Transformer translations',
    #                       title_dict={k:translator.config[k] for k in 
    #                                   ['N', 'epoch', 'num_examples']})
