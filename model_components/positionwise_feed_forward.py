import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    args: d_model - dimensions
    """

    def __init__(self, d_model, d_ff, dropout_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_layer_1  =  nn.Linear(d_model, d_ff)
        self.relu_1          =  nn.ReLU(inplace=True)
        self.dropout_layer   =  nn.Dropout(dropout_prob)
        self.linear_layer_2  =  nn.Linear(d_ff, d_model)

    def forward(self, x):
        '''
        2 linear layers with relu activation (and dropout) in between
        linear => relu => dropout => linear
        '''
        linear_1_out  =  self.linear_layer_1(x)
        relu_1_out    =  self.relu_1(linear_1_out)
        dropout_out   =  self.dropout_layer(relu_1_out)
        linear_2_out  =  self.linear_layer_2(dropout_out)
        return linear_2_out
