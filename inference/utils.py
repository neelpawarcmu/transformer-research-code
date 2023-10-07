import torch
from torchtext.data.metrics import bleu_score
from data.processors import SentenceProcessor

def greedy_decode(model, batch, vocab_tgt):
    """
    Decodes the output from the model output probabilities using a basic
    argmax function across the output
    """
    batch_size, max_padding = batch.src.shape
    bos_id, eos_id, pad_id = vocab_tgt(["<s>", "</s>", "<blank>"])
    bos_tensor = torch.full((batch_size, 1), bos_id)
    predictions_with_bos = bos_tensor
    for i in range(max_padding - 1):
        decoder_attn_mask = batch.make_decoder_attn_mask(predictions_with_bos, pad_id)
        output_probabilities = model(batch.src, predictions_with_bos, decoder_attn_mask)
        predictions = torch.argmax(output_probabilities, dim=2)
        predictions_with_bos = torch.cat([bos_tensor, predictions], dim=1)
    return predictions_with_bos


class Translate:
    def __init__(self, vocab_src, vocab_tgt):
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
    
    @classmethod
    def compute_bleu(cls, results):
        """
        Compute a bleu score for a group of sentence translations
        TODO: deprecate this
        """
        ground_truths = [[res[1].split()] for res in results] # ground truth
        predictions = [res[2].split() for res in results] # list of list of label tokens
        return bleu_score(predictions, ground_truths)
    
    def compute_batch_bleu(self, batch):
        # convert token tensor to list of sentences
        predicted_sentences = [SentenceProcessor.tokens_to_sentence(
            batch.predictions[i], self.vocab_tgt) 
            for i in range(len(batch.predictions))]
        
        # convert token tensor to list of sentences
        actual_sentences = [SentenceProcessor.tokens_to_sentence(
            batch.tgt_label[i], self.vocab_tgt) 
            for i in range(len(batch.tgt_label))]
        
        # convert to format required by bleu function
        predictions = [sentence.split() for sentence in predicted_sentences]
        ground_truths = [[sentence.split()] for sentence in actual_sentences]
        
        # compute bleu
        bleu = bleu_score(predictions, ground_truths)
        return bleu