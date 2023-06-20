import torch.nn as nn
from multi_headed_attention import MultiHeadedAttention
from sublayer import Sublayer
from positionwise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    The steps are similar, with an added cross attention component called
    'source attention' which performs attention on the encoder output
    """

    def __init__(self, h, d_model, d_ff, dropout_prob):
        super(DecoderLayer, self).__init__()

        # Create self attention sublayer:
        #   Create a closure referencing a self attention layer.
        #   The self attention module of the decoder layer uses a mask
        #   to prevent positions from attending to subsequent positions.
        self.self_attn_module = MultiHeadedAttention(h, d_model, dropout_prob)
        tokenwise_self_attn_workhorse = lambda x, mask: \
             self.self_attn_module(query =  x,
                                   key = x,
                                   value = x,
                                   attention_mask = mask)
        self.self_attn_sublayer = Sublayer(tokenwise_self_attn_workhorse, d_model, dropout_prob)

        # Create cross attention sublayer:
        #   Create a closure referencing a cross attention layer.
        #   "memory" indicates that decoder layer operates on the output embedding from the
        #   encoder stack as explained in section 3.2.3 of the paper.
        self.cross_attn_module = MultiHeadedAttention(h, d_model, dropout_prob)
        tokenwise_cross_attn_workhorse = lambda x, memory: \
             self.cross_attn_module(query =  x,
                                    key = memory,
                                    value = memory)
        self.cross_attn_sublayer = Sublayer(tokenwise_cross_attn_workhorse, d_model, dropout_prob)

        # create feedforward sublayer
        pos_ff_workhorse = PositionwiseFeedForward(d_model, d_ff, dropout_prob)
        self.pos_ff_sublayer = Sublayer(pos_ff_workhorse, d_model, dropout_prob)


    def forward(self, x, memory, decoder_attn_mask):
        "Follow Figure 1 (right) for connections."
        # self attention
        self_attn_sublayer_output = self.self_attn_sublayer(x, decoder_attn_mask)
        # cross attention
        cross_attn_sublayer_output = \
            self.cross_attn_sublayer(self_attn_sublayer_output, memory)
        # feedforward network
        pos_ff_sublayer_output = self.pos_ff_sublayer(cross_attn_sublayer_output)

        return pos_ff_sublayer_output