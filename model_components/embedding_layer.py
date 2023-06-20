import torch.nn as nn
import math

class EmbeddingLayer(nn.Module): # TODO nn.Embedding from nn.Module
    def __init__(self, vocab, d_model):
        super(EmbeddingLayer, self).__init__()
        self.lookup_table = nn.Embedding(vocab, d_model)
        self.scale_factor = math.sqrt(d_model)

    def forward(self, token): # TODO: super.forward() or similar
        # get embedding vector
        embedding_vector = self.lookup_table(token)
        # scale the vector
        scaled_embedding_vector = embedding_vector * self.scale_factor
        return scaled_embedding_vector