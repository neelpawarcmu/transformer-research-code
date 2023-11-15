import torch
from model.full_model import TransformerModel

def load_model(config, pretrained=False):
    if pretrained:
        torch.load_state_dict()
    return TransformerModel()