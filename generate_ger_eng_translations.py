import torch
import argparse
import matplotlib.pyplot as plt
from dataset_utils import load_tokenizers, load_vocab, create_dataloaders, Batch
from model.full_model import TransformerModel


def get_subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(
        torch.ones(attn_shape), diagonal=1
        ).type(torch.uint8)
    return subsequent_mask == 0

def greedy_decode(model, src, max_len, start_symbol):
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        decoder_attn_mask = get_subsequent_mask(ys.size(1)).type_as(src.data)
        prob = model(
            src, ys, decoder_attn_mask
        )[:, -1]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples,
    model_path,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    model_name, epoch_num = model_path.split(".")[0].split("/")[-1].split("_epoch_")

    print_text = ""
    print_text += f"Transformer model name: {model_name} | Epoch: {epoch_num}\n"

    for idx, batch in list(enumerate(valid_dataloader))[:n_examples]:
        rb = Batch(batch[0], batch[1], pad_idx)
        greedy_decode(model, rb.src, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        model_out = greedy_decode(model, rb.src, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )

        print_text += (
            f"\nExample {idx+1} ========\n" + 
            "Source Text (Input)        : " +
            " ".join(src_tokens).replace("\n", "") +
            "\n" + 
            "Target Text (Ground Truth) : " +
            " ".join(tgt_tokens).replace("\n", "") +
            "\n" + 
            "Model Output               : " +
            model_txt.replace("\n", "") + 
            "\n"
        )

        print(print_text)

        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results, print_text

def run_model_example(vocab_src, vocab_tgt, spacy_de, spacy_en, model_path, output_path, n_examples=5):

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(torch.device("cpu"), vocab_src, 
                                             vocab_tgt, spacy_de, spacy_en, 
                                             batch_size=1, shuffle=False)

    print("Loading Trained Model ...")
    model = TransformerModel(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data, print_text = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples, 
        model_path
    )

    save_translations(print_text, save_path=output_path)
    return model, example_data

def save_translations(print_text, save_path):
    """
    Save translation text as an image
    """
    plt.figure()
    plt.text(0, 1, print_text)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", type=int, default=1) # 1-indexed epoch number of saved model
    parser.add_argument("--model_save_name", type=str, 
                        default="artifacts/saved_models/multi30k_model")
    parser.add_argument("--output_save_name", type=str, 
                        default="artifacts/generated_translations/translation")
    args = parser.parse_args()
    model_path = f"{args.model_save_name}_epoch_{args.model_epoch}.pt"
    output_path = f"{args.output_save_name}_epoch_{args.model_epoch}.png"

    # load vocabulary
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    run_model_example(vocab_src, vocab_tgt, spacy_de, spacy_en, model_path, output_path)