from dotted_dict import DottedDict
import yaml
import json
import os
from typing import Any

class Config(DottedDict):
    """
    Configuration class that extends DottedDict with loading/saving functionality
    """
    
    @classmethod
    def load(cls, config_path: str = "configs/default.yaml") -> "Config":
        """Load YAML configuration file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def update_from_args(self, args: Any) -> "Config":
        """Update config with command line arguments"""
        arg_to_config_mapping = {
            'N': 'model.n_layers',
            'epochs': 'training.epochs',
            'batch_size': 'training.batch_size',
            'dataset_name': 'dataset.name',
            'dataset_size': 'dataset.size',
            'max_padding': 'model.max_padding',
            'cache': 'dataset.cache',
            'tokenizer_type': 'tokenizer.type',
            'random_seed': 'training.random_seed'
        }
        for arg_name, config_path in arg_to_config_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                # Split the config path and set nested value
                keys = config_path.split('.')
                current = self
                for key in keys[:-1]:
                    current = current[key]
                current[keys[-1]] = getattr(args, arg_name)
        return self
    
    def save(self, save_path: str, format: str = "yaml"):
        """Save config to YAML file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if format == "yaml":
            with open(save_path, 'w') as f:
                yaml.dump(dict(self), f)
        elif format == "json":
            with open(save_path, 'w') as f:
                json.dump(dict(self), f)