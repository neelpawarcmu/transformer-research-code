import torch.nn as nn
from torch.nn.functional import log_softmax

class LinearSoftmaxLayer(nn.Module):
    "Define standard linear + softmax generation step."
    """
    1. projecting
    2. generating probabilities
    """

    def __init__(self, d_model, vocab):
        super(LinearSoftmaxLayer, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)