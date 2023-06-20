import argparse
from model.full_model import TransformerModel

def create_model(src_vocab_size: int,
                 tgt_vocab_size: int,
                 N: int = 6, 
                 d_model: int = 512,
                 d_ff: int = 2048,
                 h: int = 8, 
                 dropout_prob: float = 0.1):
    model = TransformerModel(src_vocab_size, tgt_vocab_size, N, d_model, d_ff, h, dropout_prob)
    print(model)
    return model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_vocab_size", type=int, default=11) 
    parser.add_argument("--tgt_vocab_size", type=int, default=11) 
    parser.add_argument("--N", type=int, default=6) 
