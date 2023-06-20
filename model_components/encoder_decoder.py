import torch.nn as nn
from encoder_stack import EncoderStack
from decoder_stack import DecoderStack

class EncoderDecoder(nn.Module):
    """
    A standard baseline Encoder-Decoder architecture. Base for this and many
    other models. At each step, model is auto regressive model, consuming
    the previously generated symbols as additional input when generating
    the next.
    """

    def __init__(self, h, d_model, d_ff, dropout_prob, N):
        super(EncoderDecoder, self).__init__()
        self.encoder_stack = EncoderStack(h, d_model, d_ff, dropout_prob, N)
        self.decoder_stack = DecoderStack(h, d_model, d_ff, dropout_prob, N)