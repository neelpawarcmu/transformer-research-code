from model.full_model import TransformerModel
from numerize.numerize import numerize as nu

def load_model(config, pretrained=False, model_path=None):
    model =  TransformerModel(config["src_vocab_size"], 
                              config["tgt_vocab_size"], 
                              config["N"], 
                              config["d_model"], 
                              config["d_ff"], 
                              config["h"], 
                              config["dropout_prob"])
    if pretrained:
        model.load_state_dict(model_path)
    return model

def count_params(model):
    num_param_matrices, num_params, num_trainable_params = 0, 0, 0
    
    for p in model.parameters():
        num_param_matrices += 1
        num_params += p.numel()
        num_trainable_params += p.numel() * p.requires_grad

    print(f"{'-'*60}"
          f"\n{model.__class__.__name__}\n"
          f"Number of parameter matrices: {nu(num_param_matrices)}\n"
          f"Number of parameters: {nu(num_params)}\n"
          f"Number of trainable parameters: {nu(num_trainable_params)}")
    return num_param_matrices, num_params, num_trainable_params