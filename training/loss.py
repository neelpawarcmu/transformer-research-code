import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    """
    Implement label smoothing. TODO: improve this docstring
    """
    def __init__(self, vocab_size, pad_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.vocab_size
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        mask = torch.nonzero(target.data == self.pad_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class SimpleLossCompute:
    """
    A simple loss compute and train function.
    """
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, y_preds, y_labels, batch_ntokens):
        loss = self.criterion(
                y_preds.contiguous().view(-1, y_preds.size(-1)), 
                y_labels.contiguous().view(-1)
            ) / batch_ntokens
        return loss