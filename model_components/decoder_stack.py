import torch.nn as nn
from decoder_layer import DecoderLayer

class DecoderStack(nn.Module):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, h, d_model, d_ff, dropout_prob, N):
        super(DecoderStack, self).__init__()

        # create and stack decoder layers
        decoder_layer_list = []
        for i in range(N):
            decoder_layer_i = DecoderLayer(h, d_model, d_ff, dropout_prob)
            decoder_layer_list.append(decoder_layer_i)
        self.decoder_layers = nn.ModuleList(decoder_layer_list)

    def forward(self, x, memory, decoder_attn_mask):
        layer_input = x
        for layer in self.decoder_layers:
            # compute layer output
            layer_output = layer(layer_input, memory, decoder_attn_mask)
            # this becomes input to next layer
            layer_input = layer_output

        return layer_output