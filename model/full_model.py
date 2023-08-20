import math
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048,
                 h=8, dropout_prob=0.1):
        '''
        This class assembles the transformer model from the individual submodules created,
        block by block as shown in Figure 1 of the paper.
           src_vocab_size: number of tokens in encoder's embedding dictionary
           tgt_vocab_size: number of tokens in decoder's embedding dictionary
           N: number of encoder/decoder layers
           d_model: embedding size
           d_ff: feedfirward layer size
           h: number of attention heads
        '''
        super(TransformerModel, self).__init__()

        # embedding layers
        self.input_embedding_layer = EmbeddingLayer(src_vocab_size, d_model)
        self.output_embedding_layer = EmbeddingLayer(tgt_vocab_size, d_model)

        # positional encoding layers
        self.input_positional_enc_layer = PositionalEncodingLayer(d_model, dropout_prob)
        self.output_positional_enc_layer = PositionalEncodingLayer(d_model, dropout_prob)

        # encoder-decoder
        self.encoder_decoder = EncoderDecoder(h, d_model, d_ff, dropout_prob, N)

        # linear and softmax layers
        self.linear_softmax_layers = LinearSoftmaxLayer(d_model, tgt_vocab_size)

        # Initialize parameters with Glorot / fan_avg.
        # This was important according to the paper's code TODO: verify this from the code
        for p in self.parameters():
            if p.dim() > 1: # presumably biases skipped TODO: verify this
                nn.init.xavier_uniform_(p)

    def encode(self, src):
        # embed and add positional encoding
        input_embeddings = self.input_embedding_layer(src)
        input_embeddings_with_positions = self.input_positional_enc_layer(input_embeddings)
        # encode
        encoder_stack_output = self.encoder_decoder.encoder_stack(input_embeddings_with_positions)
        return encoder_stack_output

    def decode(self, tgt, memory, decoder_attn_mask):
        # embed and add positional encoding
        output_embeddings = self.output_embedding_layer(tgt)
        output_embeddings_with_positions = self.output_positional_enc_layer(output_embeddings)
        # decode
        decoder_stack_output = self.encoder_decoder.decoder_stack(output_embeddings_with_positions, memory, decoder_attn_mask)
        return decoder_stack_output

    def forward(self, src, tgt, decoder_attn_mask):
        encoder_stack_output = self.encode(src)
        decoder_stack_output = self.decode(tgt, encoder_stack_output, decoder_attn_mask)
        output_probabilities = self.linear_softmax_layers(decoder_stack_output)

        return output_probabilities


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


class PositionalEncodingLayer(nn.Module):
    """
    Implement the PE function."""

    def __init__(self, d_model, dropout_prob, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout_prob)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x_with_position = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout_layer(x_with_position)


class EncoderDecoder(nn.Module):
    """
    A standard baseline Encoder-Decoder architecture. Base for this and many
    other models. At each step, model is auto regressive model, consuming
    the previously generated symbols as additional input when generating
    the next.
    """

    def __init__(self, h, d_model, d_ff, dropout_prob, N):
        super(EncoderDecoder, self).__init__()
        self.encoder_stack = EncoderStack(h, d_model, d_ff, dropout_prob, N)
        self.decoder_stack = DecoderStack(h, d_model, d_ff, dropout_prob, N)


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

        # layer normalization
        self.norm_layer = LayerNorm(d_model)

    def forward(self, x):
        "Pass the input through each layer in turn."
        layer_input = x
        for layer in self.encoder_layers:
            # compute layer output
            layer_output = layer(layer_input)
            # this becomes input to next layer
            layer_input = layer_output

        normed_output = self.norm_layer(layer_output)
        return normed_output


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

        # layer normalization
        self.norm_layer = LayerNorm(d_model)

    def forward(self, x, memory, decoder_attn_mask):
        layer_input = x
        for layer in self.decoder_layers:
            # compute layer output
            layer_output = layer(layer_input, memory, decoder_attn_mask)
            # this becomes input to next layer
            layer_input = layer_output

        normed_output = self.norm_layer(layer_output)
        return normed_output


class Sublayer(nn.Module):
    """
    Sublayer accepts a workhorse (either a closure with a self attention
    layer inside, or a position-wise feed forward layer)
    as an argument and composes it with the following operations:
    x + Dropout(Workhorse(LayerNorm(x)))
    """
    def __init__(self, workhorse, size, dropout_prob):
        super(Sublayer, self).__init__()
        self.workhorse = workhorse
        self.norm_layer = LayerNorm(size)
        self.dropout_layer = nn.Dropout(p = dropout_prob)

    def forward(self, x, mask=None, memory=None): # x is representation / embedding
        normed_x = self.norm_layer(x)
        if mask is not None: # decoder self attention sublayer
            workhorse_output = self.workhorse(normed_x, mask)
        elif memory is not None: # decoder cross attention sublayer
            workhorse_output = self.workhorse(normed_x, memory)
        else: # encoder or feedforward sublayer
            workhorse_output = self.workhorse(normed_x)
        # apply dropout
        dropout_output = self.dropout_layer(workhorse_output)
        # residual connection
        residual_output = x + dropout_output
        return residual_output

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features))
        self.b_2 = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    """
    EncoderLayer is made up of one self-attention sublayer and
    one positionwise feed forward sublayer
    """

    def __init__(self, h, d_model, d_ff, dropout_prob):
        super(EncoderLayer, self).__init__()

        # Initialize self attention network
        # note that we save as class variable (self.) only for torch functionality
        self.self_attn_module = MultiHeadedAttention(h, d_model, dropout_prob)

        # Create self attention sublayer:
        #   Create a closure with self attention module inside.
        #   Encoder layers do not use the memory argument, because k and v are
        #   the input from the previous layer
        tokenwise_self_attn_workhorse = lambda x: \
             self.self_attn_module(query =  x,
                                   key = x,
                                   value = x)
        self.self_attn_sublayer = Sublayer(tokenwise_self_attn_workhorse, d_model, dropout_prob)

        # create feedforward sublayer
        pos_ff_workhorse = PositionwiseFeedForward(d_model, d_ff, dropout_prob)
        self.pos_ff_sublayer = Sublayer(pos_ff_workhorse, d_model, dropout_prob)

    def forward(self, x):
        self_attn_sublayer_output = self.self_attn_sublayer(x)
        pos_ff_sublayer_output = self.pos_ff_sublayer(self_attn_sublayer_output)

        return pos_ff_sublayer_output

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
        reshape_fn = lambda x : x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        derived_queries = reshape_fn(self.w_q(query))
        derived_keys = reshape_fn(self.w_k(key))
        derived_values = reshape_fn(self.w_v(value))

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


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    args: d_model - dimensions
    """

    def __init__(self, d_model, d_ff, dropout_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_layer_1  =  nn.Linear(d_model, d_ff)
        self.relu_1          =  nn.ReLU(inplace=True)
        self.linear_layer_2  =  nn.Linear(d_ff, d_model)
        self.dropout_layer   =  nn.Dropout(dropout_prob)

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


class LinearSoftmaxLayer(nn.Module):
    "Define standard linear + softmax generation step."
    """
    1. projecting
    2. generating probabilities
    """

    def __init__(self, d_model, vocab_size):
        super(LinearSoftmaxLayer, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

