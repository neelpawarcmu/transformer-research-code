from transformers import AutoTokenizer

# class BPETokenizer:
#     def __init__(self, language):
#         self.language = language
#         self.available_pipelines = {"en": "bert-base-cased",
#                                     "de": "dbmdz/bert-base-german-cased"}
#         self.load_bpe_pipeline()
    
#     def load_bpe_pipeline(self):
#         pipeline_name = self.available_pipelines[self.language]
#         self.bpe_pipeline = AutoTokenizer.from_pretrained(pipeline_name)
    
#     def tokenize(self, text):
#         # return [tok.text for tok in self.bpe_pipeline.tokenizer(text)]

def build_tokenizers(language_pair):
    available_tokenizers = {"en": "bert-base-cased",
                            "de": "dbmdz/bert-base-german-cased"}
    tokenizer_src = AutoTokenizer.from_pretrained(available_tokenizers[language_pair[0]])
    tokenizer_tgt = AutoTokenizer.from_pretrained(available_tokenizers[language_pair[1]])
    tokenizer_src.add_special_tokens({"bos_token":"<s>",
                                      "eos_token":"</s>"})
    tokenizer_tgt.add_special_tokens({"bos_token":"<s>",
                                      "eos_token":"</s>"})
    return tokenizer_src, tokenizer_tgt