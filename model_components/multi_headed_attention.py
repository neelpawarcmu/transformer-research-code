import torch
import torch.nn as nn
import math

def attention_fn(derived_queries, derived_keys, derived_values, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # key size
    d_k = derived_queries.size(-1)
    # equation (1) of paper
    scores = torch.matmul(derived_queries, derived_keys.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weightings = scores.softmax(dim=-1)
    if dropout is not None:
        attention_weightings = dropout(attention_weightings)
    return torch.matmul(attention_weightings, derived_values), attention_weightings


class MultiHeadedAttention(nn.Module):
    """
    Generates a multiheaded attention network, which is 3 feedforward networks,
    for linear transformation on the query, key and value.
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super().__init__()
        # @?? ensure dimensionality of model ie. output of multihead attention
        # matches number of heads * individual attention head dimension
        assert d_model % h == 0, f'dimension mismatch, d_model must be a multiple of h (got {d_model} and {h})'
        self.h = h # number of heads
        self.d_k = d_model // h # assume size_key = size_val

        # create linear layers for weights corresponding to q, k, v
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attention_mask=None):
        # TODO: explain what is this doing
        if attention_mask is not None:
            attention_mask_tensor = attention_mask.unsqueeze(1)
        else:
            attention_mask_tensor = None

        batch_size = query.size(0)

        # TODO: comment here what we are doing, explain reshaping operations
        # reshape outputs to @@@ shape (TODO)
        derived_queries = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # TODO: reshape better?
        derived_keys = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        derived_values = self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # compute attention
        attention_outputs, attention_weightings = \
            attention_fn(derived_queries, derived_keys, derived_values,
                         attention_mask_tensor, dropout=self.dropout_layer)
        # save weightings for visualization
        self.attention_weightings = attention_weightings
        # reshape attention outputs to @@@ (TODO)
        reshaped_attention_outputs = attention_outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # pass through final linear layer
        result = self.linear_layer(reshaped_attention_outputs)
        del query, key, value, derived_queries, derived_keys, derived_values

        return result