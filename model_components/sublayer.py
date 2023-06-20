import torch.nn as nn
from layer_norm import LayerNorm

class Sublayer(nn.Module):
    """
    Sublayer accepts a workhorse (either a closure with a self attention
    layer inside, or a position-wise feed forward layer)
    as an argument and composes it with the following operations:
    LayerNorm(x + Dropout(Workhorse(x)))
    """
    def __init__(self, workhorse, size, dropout_prob):
        super(Sublayer, self).__init__()
        self.workhorse = workhorse
        self.norm_layer = LayerNorm(size)
        self.dropout_layer = nn.Dropout(p = dropout_prob)

    def forward(self, x, mask=None, memory=None): # x is representation / embedding
        if mask is not None: # decoder self attention sublayer
            workhorse_output = self.workhorse(x, mask)
        elif memory is not None: # decoder cross attention sublayer
            workhorse_output = self.workhorse(x, memory)
        else: # encoder or feedforward sublayer
            workhorse_output = self.workhorse(x)
        # apply dropout
        dropout_output = self.dropout_layer(workhorse_output)
        # residual connection
        residual_output = x + dropout_output
        # layer norm
        sublayer_output = self.norm_layer(residual_output)
        return sublayer_output