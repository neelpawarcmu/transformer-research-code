import json
import argparse
import torch
from vocab.vocab_utils import build_tokenizers, load_vocabularies
from data.runtime_loaders import load_datasets, load_dataloaders
from data.processors import SentenceProcessor
from model.full_model import TransformerModel
from training.logging import DirectoryCreator, TranslationLogger
from inference.utils import greedy_decode, BleuUtils
from data.download import DataDownloader

class Translator:
    def __init__(self, args, config_path):
        '''
        Initializes the Translator class by creating required directories 
        and loading the runtime configs
        '''
        # load model and training configurations saved from training run
        self.load_config(config_path, args)
        # create directories required for saving artifacts
        DirectoryCreator.add_dir(f"generated_translations/N{self.config['N']}", 
                                 include_base_path=True)
    
    def load_config(self, filepath, args):
        '''
        Load a saved configuration json file as a dictionary to be used 
        for model loading and translation.
        '''
        with open(filepath, 'r') as fp:
            config = json.load(fp)
        # add information from args
        config["epoch"] = args.epoch
        config["num_examples"] = args.num_examples
        config["N"] = args.N
        self.config = config

    def prepare_vocabs(self): 
        '''
        Load tokenizers and vocabularies
        '''
        tokenizer_src, tokenizer_tgt = build_tokenizers(self.config['language_pair'])
        vocab_src, vocab_tgt = load_vocabularies(cache=True)
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt


    def prepare_model(self):
        # create model structure
        config = self.config
        self.model = TransformerModel(config['src_vocab_size'],
                                      config['tgt_vocab_size'],
                                      config['N'],
                                      config['d_model'],
                                      config['d_ff'],
                                      config['h'],
                                      config['dropout_prob'])
                                      
        # load saved weights on to the model
        save_path = f"{config['model_dir']}/N{config['N']}/epoch_{config['epoch']:02d}.pt"
        self.model.load_state_dict(torch.load(save_path))

    def prepare_dataloader(self, split):
        train_dataset, val_dataset, test_dataset = load_datasets(
            self.config["dataset_name"],
            self.config["language_pair"], 
            self.tokenizer_src, 
            self.tokenizer_tgt, 
            self.vocab_src,
            self.vocab_tgt,
            self.config["max_padding"],
            device=torch.device("cpu"),
            cache=True,
            random_seed=4)

        train_dataloader, val_dataloader, test_dataloader = load_dataloaders(
            train_dataset, 
            val_dataset, 
            test_dataset,
            self.config["batch_size"],
            shuffle=False)
        
        # select dataloader to use for translation
        self.dataloader = eval(f'{split}_dataloader')

    def translate(self):
        for batch in list(self.dataloader)[:self.config['num_examples']]:
            batch.predictions = greedy_decode(self.model, batch, self.vocab_tgt)
            src_sentence = SentenceProcessor.tokens_to_sentence(batch.src[0], self.vocab_src)
            tgt_sentence = SentenceProcessor.tokens_to_sentence(batch.tgt_shifted_right[0], self.vocab_tgt)
            pred_sentence = SentenceProcessor.tokens_to_sentence(batch.predictions[0], self.vocab_tgt)
            logger.log_sentence('Source sentence', src_sentence)
            logger.log_sentence('Target sentence (Ground truth)', tgt_sentence)
            logger.log_sentence('Predicted sentence (Model output)', pred_sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1
                        ) # 1-indexed epoch number of saved model
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--split", type=str, default='test', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    # initialize Translator class
    translator = Translator(args, 'artifacts/training_config.json')
    
    # initialize logger
    logger = TranslationLogger()

    # prepare fundamental blocks of translation pipeline
    translator.prepare_vocabs()
    translator.prepare_dataloader(split=args.split)
    translator.prepare_model()

    # run translations
    translator.translate()

    # print and save translation results
    logger.print_and_save('artifacts/generated_translations/', 
                          title='Transformer translations',
                          title_dict={k:translator.config[k] for k in 
                                      ['N', 'epoch', 'num_examples']})
