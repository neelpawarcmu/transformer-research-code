import torch
import torch.nn as nn
import math
import copy
from torch.nn.functional import log_softmax
from model.utils import count_params

class EncoderDecoder(nn.Module):
    """
    A standard baseline Encoder-Decoder architecture. Base for this and many
    other models. At each step, model is auto regressive model, consuming
    the previously generated symbols as additional input when generating
    the next.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        logprobs = self.decode( # Refactored NP
            self.encode(
                src, src_mask
                ),
                src_mask, tgt, tgt_mask)
        return self.generator(logprobs)

    def encode(self, src, src_mask):
        """
        symbol representations -> continuous representations
        embed -> encode
        """
        return self.encoder(
            self.src_embed(src), src_mask
            )

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        continuous representations -> symbol representations
        embed -> decode
        """
        return self.decoder(
            self.tgt_embed(tgt), memory, src_mask, tgt_mask
            )

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, encoder_layer, N):
        super(Encoder, self).__init__()
        self.encoder_layer_1 = copy.deepcopy(encoder_layer)
        self.encoder_layer_2 = copy.deepcopy(encoder_layer)
        self.encoder_layer_3 = copy.deepcopy(encoder_layer)
        self.encoder_layer_4 = copy.deepcopy(encoder_layer)
        self.encoder_layer_5 = copy.deepcopy(encoder_layer)
        self.encoder_layer_6 = copy.deepcopy(encoder_layer)
        self.norm_layer = LayerNorm(encoder_layer.d_model)

    def forward(self, x, mask): #@@today better term for x?
        "Pass the input (and mask) through each layer in turn."
        enc_layer_1_out = self.encoder_layer_1(x, mask)
        enc_layer_2_out = self.encoder_layer_2(enc_layer_1_out, mask)
        enc_layer_3_out = self.encoder_layer_3(enc_layer_2_out, mask)
        enc_layer_4_out = self.encoder_layer_4(enc_layer_3_out, mask)
        enc_layer_5_out = self.encoder_layer_5(enc_layer_4_out, mask)
        enc_layer_6_out = self.encoder_layer_6(enc_layer_5_out, mask)

        normed_out = self.norm_layer(enc_layer_6_out)
        return normed_out
    

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class Sublayer(nn.Module):
    """
    Sublayer accepts a workhorse layer (either attention head or fully connected layer)
    as an argument and performs the following operation on it:
    LayerNorm(x + Dropout(Workhorse(x)))
    """
    def __init__(self, size, dropout_prob):
        super(Sublayer, self).__init__()
        self.norm_layer = LayerNorm(size)
        self.dropout_layer = nn.Dropout(p = dropout_prob)

    def forward(self, x, workhorse_layer): # x is representation / embedding
        self.workhorse_layer = workhorse_layer
        workhorse_output = self.workhorse_layer(self.norm_layer(x))
        # apply dropout
        dropout_output = self.dropout_layer(workhorse_output)
        # residual connection
        residual_output = x + dropout_output
        # layer norm
        sublayer_output = residual_output
        return sublayer_output
    

class EncoderLayer(nn.Module):
    """
    EncoderLayer is made up of one self-attention layer and one positionwise feed forward layer
    """

    def __init__(self, d_model, self_attn_layer, pos_ff_layer, dropout_prob):
        '''
        Initialize attention sublayer and positionwise feedforward sublayer
        These are identically initialized but will differ in the forward method

        Also save self attention and pos ff layers for passing later as args to
        forward method
        '''
        super(EncoderLayer, self).__init__()

        # save layers for args
        self.self_attn_layer  =  self_attn_layer # TODO: create MHAttention here
        self.pos_ff_layer     =  pos_ff_layer

        # init sublayers
        self.self_attn_sublayer =  Sublayer(d_model, dropout_prob) # TODO: pass MHAttn
        self.pos_ff_sublayer    =  Sublayer(d_model, dropout_prob)

        # save size
        self.d_model = d_model

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections.
        """
        # map attention fn between current token and every token in sentence
        tokenwise_self_attn_fn = lambda x: self.self_attn_layer(query =  x,
                                                                key   =  x,
                                                                value =  x,
                                                                attention_mask = mask)
        # x => self attention sub layer => attention_out
        self_attn_sublayer_out  =  self.self_attn_sublayer(x, tokenwise_self_attn_fn)

        # attention_out => positionwise ff sub layer => pos_ff_out
        pos_ff_sublayer_out     =  self.pos_ff_sublayer(self_attn_sublayer_out, self.pos_ff_layer)

        return pos_ff_sublayer_out
    


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, decoder_layer, N):
        super(Decoder, self).__init__()
        self.decoder_layers = clones(decoder_layer, N)
        self.norm_layer     = LayerNorm(decoder_layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, src_mask, tgt_mask)
        decoder_out = self.norm_layer(x)
        return decoder_out
    

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    The steps are similar, with only an added cross attention component called
    'source attention' which performs attention on the encoder output
    """

    def __init__(self, size, self_attn_layer, src_attn_layer, pos_ff_layer, dropout_prob):
        super(DecoderLayer, self).__init__()

        # init sublayers
        self.self_attn_sublayer =  Sublayer(size, dropout_prob)
        self.src_attn_sublayer  =  Sublayer(size, dropout_prob)
        self.pos_ff_sublayer    =  Sublayer(size, dropout_prob)

        # save layers for args
        # @today: better names for these
        self.self_attn_layer  =  self_attn_layer
        self.src_attn_layer   =  src_attn_layer
        self.pos_ff_layer     =  pos_ff_layer

        # save size
        self.size = size


    def forward(self, token_vector, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory # TODO: explain what this is doing in this comment

        # self attention and subsequent layernorm
        # @today: better name for tokenwise_self_attn_fn & tokenwise_self_attn_fn
        tokenwise_self_attn_fn = lambda x: self.self_attn_layer(query = x,
                                                                key   = x,
                                                                value = x,
                                                                attention_mask = tgt_mask)

        tokenwise_src_attn_fn  = lambda x: self.src_attn_layer(query = x,
                                                               key   = m,
                                                               value = m,
                                                               attention_mask = src_mask)
        # token_vector => self attention sub layer => attention_out
        self_attn_out  =  self.self_attn_sublayer(token_vector, tokenwise_self_attn_fn)
        # token_vector => self attention sub layer => attention_out
        src_attn_out   =  self.src_attn_sublayer(self_attn_out, tokenwise_src_attn_fn)
        # token_vector => self attention sub layer => attention_out
        pos_ff_out     =  self.pos_ff_sublayer(src_attn_out, self.pos_ff_layer)

        return pos_ff_out
    

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(
        torch.ones(attn_shape), diagonal=1
        ).type(torch.uint8)
    return subsequent_mask == 0

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
    TODO: purpose of this class: what does this layer do? i/ps o/ps?
    From the attention equation,
    d_k =
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
        self.w_q     =  nn.Linear(d_model, d_model)
        self.w_k     =  nn.Linear(d_model, d_model)
        self.w_v     =  nn.Linear(d_model, d_model)
        self.linear_layer  =  nn.Linear(d_model, d_model)
        self.dropout_layer =  nn.Dropout(p=dropout)

    def forward(self, query, key, value, attention_mask=None):
        # @?? what is this doing?
        if attention_mask is not None:
            attention_mask_tensor = attention_mask.unsqueeze(1)
        else:
            attention_mask_tensor = None

        batch_size = query.size(0)

        # TODO: comment here what we are doing, explain reshaping operations
        # reshape outputs to @@@ shape
        derived_queries =  self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # @?? reshape better?
        derived_keys    =  self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        derived_values  =  self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # compute attention
        attention_outputs, attention_weightings = \
            attention_fn(derived_queries, derived_keys, derived_values,
                         attention_mask_tensor, dropout=self.dropout_layer)
        # save weightings for visualization
        self.attention_weightings = attention_weightings
        # reshape attention outputs to @@@
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
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lookup_table = nn.Embedding(vocab, d_model)
        self.scale_factor = math.sqrt(d_model)

    def forward(self, token):
        token_vector         = self.lookup_table(token)
        scaled_token_vector  = token_vector * self.scale_factor
        return scaled_token_vector
    

class PositionalEncoding(nn.Module):
    """
    @today: name suggestions?
    Implement the PE function."""

    def __init__(self, d_model, dropout_prob, max_len=5000):
        super(PositionalEncoding, self).__init__()
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
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout_layer(x)
    

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout_prob=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy

    # @@today is attn_network ok or attn_heads?
    attn_network      = MultiHeadedAttention(h, d_model)
    pos_ff_network    = PositionwiseFeedForward(d_model, d_ff, dropout_prob)
    pos_enc_network   = PositionalEncoding(d_model, dropout_prob)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn_network), c(pos_ff_network), dropout_prob), N),
        Decoder(DecoderLayer(d_model, c(attn_network), c(attn_network), c(pos_ff_network), dropout_prob), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(pos_enc_network)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(pos_enc_network)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    count_params(model)
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(config, save_path):
    model = make_model(config["src_vocab_size"], config["tgt_vocab_size"])
    model = model.load_state_dict(torch.load(save_path))
    model = model.to(config["device"])
    return model