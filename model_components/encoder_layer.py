import torch.nn as nn
from multi_headed_attention import MultiHeadedAttention
from sublayer import Sublayer
from positionwise_feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    EncoderLayer is made up of one self-attention sublayer and
    one positionwise feed forward sublayer
    """

    def __init__(self, h, d_model, d_ff, dropout_prob):
        super(EncoderLayer, self).__init__()

        # Initialize self attention network
        # note that we save as class variable (self.) only for torch functionality
        self.self_attn_module = MultiHeadedAttention(h, d_model, dropout_prob)

        # Create self attention sublayer:
        #   Create a closure with self attention module inside.
        #   Encoder layers do not use the memory argument, because k and v are
        #   the input from the previous layer
        tokenwise_self_attn_workhorse = lambda x: \
             self.self_attn_module(query =  x,
                                   key = x,
                                   value = x)
        self.self_attn_sublayer = Sublayer(tokenwise_self_attn_workhorse, d_model, dropout_prob)

        # create feedforward sublayer
        pos_ff_workhorse = PositionwiseFeedForward(d_model, d_ff, dropout_prob)
        self.pos_ff_sublayer = Sublayer(pos_ff_workhorse, d_model, dropout_prob)

    def forward(self, x):
        self_attn_sublayer_output = self.self_attn_sublayer(x)
        pos_ff_sublayer_output = self.pos_ff_sublayer(self_attn_sublayer_output)

        return pos_ff_sublayer_output