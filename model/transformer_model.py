import torch.nn as nn
from model_components.embedding_layer import EmbeddingLayer
from model_components.positional_encoding_layer import PositionalEncodingLayer
from model_components.encoder_decoder import EncoderDecoder
from model_components.linear_softmax_layer import LinearSoftmaxLayer

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048,
                 h=8, dropout_prob=0.1):
        '''
        This class assembles the transformer model from the submodules created above,
        block by block as shown in Figure 1 of the paper.
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