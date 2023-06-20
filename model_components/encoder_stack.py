import torch.nn as nn
from encoder_layer import EncoderLayer

class EncoderStack(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, h, d_model, d_ff, dropout_prob, N):
        super(EncoderStack, self).__init__()

        # create and stack encoder layers
        encoder_layer_list = []
        for i in range(N):
            encoder_layer_i = EncoderLayer(h, d_model, d_ff, dropout_prob)
            encoder_layer_list.append(encoder_layer_i)
        self.encoder_layers = nn.ModuleList(encoder_layer_list)

    def forward(self, x):
        "Pass the input through each layer in turn."
        layer_input = x
        for layer in self.encoder_layers:
            # compute layer output
            layer_output = layer(layer_input)
            # this becomes input to next layer
            layer_input = layer_output

        return layer_output