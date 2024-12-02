import os
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from numpy import pad
import numpy as np
import matplotlib.pyplot as plt # TODO: remove this
from collections import defaultdict
import json

class DataProcessor:
    def __init__(self, tokenizer_src, tokenizer_tgt,
                 max_padding, language_pair):
        super().__init__()
        # TODO: language_pair arg comes first for consistency
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_padding = max_padding
        self.language_pair = language_pair

    def plot_token_lengths(self, raw_data):
        fig, axs = plt.subplots(nrows=len(raw_data.keys()), ncols=len(self.language_pair), 
                            figsize=(15, 5*len(raw_data.keys())), dpi=500)
        
        # Make axs 2D if there's only one language pair
        if len(self.language_pair) == 1:
            axs = np.array([axs]).reshape(-1, 1)

        # Load or calculate token lengths
        if os.path.exists("all_lens.json"):
            all_lens = json.load(open("all_lens.json", "r"))
        else:
            all_lens = defaultdict(lambda: defaultdict(list))
            for split in tqdm(raw_data.keys(), desc="Calculating token lengths", position=0, leave=True):
                for j, (language, tokenizer) in enumerate(zip(self.language_pair, [self.tokenizer_src, self.tokenizer_tgt])):
                    batch_size = 1000000
                    src_sentences = [src for src, _ in raw_data[split]]
                    tgt_sentences = [tgt for _, tgt in raw_data[split]]
                    if j == 0:
                        sentences = src_sentences
                    else:
                        sentences = tgt_sentences
                    for start_idx in range(0, len(sentences), batch_size):
                        end_idx = min(start_idx + batch_size, len(sentences))
                        batch_data = sentences[start_idx:end_idx]
                        tokenized_data = tokenizer(batch_data)["input_ids"]
                        all_lens[language][split].extend([len(sent) for sent in tokenized_data])
            json.dump(all_lens, open("all_lens.json", "w"))

        # Plot the distributions
        # Calculate global max_len across all splits and languages
        global_max_len = 0
        for language in self.language_pair:
            for split in raw_data.keys():
                lens = all_lens[language][split]
                length_threshold = np.percentile(lens, 99.99)
                filtered_lens = [l for l in lens if l <= length_threshold]
                split_max_len = int(np.percentile(filtered_lens, 99.99))
                global_max_len = max(global_max_len, split_max_len)
        print(f"Global max length: {global_max_len}")

        for i, language in tqdm(enumerate(self.language_pair), desc="Plotting token lengths", position=0, leave=True):
            for j, split in enumerate(raw_data.keys()):
                lens = all_lens[language][split]
                
                # Remove extreme outliers (beyond 99.99 percentile)
                length_threshold = np.percentile(lens, 99.99)
                filtered_lens = [l for l in lens if l <= length_threshold]
                
                # Calculate statistics
                quartiles_of_interest = [25, 50, 75, 90, 95, 99]
                quartiles = np.percentile(filtered_lens, quartiles_of_interest)
                mean = np.mean(filtered_lens)
                
                # Plot histogram with better binning
                # max_len = int(np.percentile(filtered_lens, 99.99))
                bins = np.linspace(0, global_max_len, 50)
                
                counts, bins, _ = axs[j, i].hist(filtered_lens, bins=bins, alpha=0.7)
                axs[j, i].grid(True, alpha=0.3)
                axs[j, i].set_title(f"{language} - {split} (Total: {len(filtered_lens):,} sentences)")
                axs[j, i].set_xlabel("Token Length")
                axs[j, i].set_ylabel("Number of Sentences")
                
                # Format y-axis with comma separator
                axs[j, i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                
                # Add vertical lines for statistics
                values = [*quartiles, mean]
                labels = [f'{q}th percentile' for q in quartiles_of_interest] + ['Mean']
                colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black']
                
                # Sort by value while keeping labels aligned
                sorted_indices = np.argsort(values)
                sorted_values = np.array(values)[sorted_indices]
                sorted_labels = np.array(labels)[sorted_indices]
                sorted_colors = np.array(colors)[sorted_indices]
                
                for label, value, color in zip(sorted_labels, sorted_values, sorted_colors):
                    axs[j, i].axvline(value, linestyle='--', label=f'{label}: {int(value)}', color=color)
                
                axs[j, i].legend(fontsize='small')
                print(f"\n{language} - {split}:")
                print(f"Total sentences: {len(filtered_lens):,}")
                print(f"Filtered out: {len(lens) - len(filtered_lens):,} sentences beyond {int(length_threshold)} tokens")

        plt.tight_layout()
        plt.suptitle(f"Token count statistics for '{self.language_pair}' language pair", fontsize=16, y=1.05, fontweight='bold')
        plt.savefig(f"artifacts/token_lengths_{'-'.join(self.language_pair)}.png")
        import pdb; pdb.set_trace()

    def preprocess_data(self, raw_data):
        '''
        Preprocess raw data sentence by sentence and save to disk.
        Returns: Dict of torch tensors for each split
        '''
        preprocessed_dataset = {}
        for split in tqdm(raw_data.keys(), desc="Preprocessing dataset", position=0, leave=True):
            # Process in smaller batches to manage memory
            batch_size = 10000
            src_sentences = [src for src, tgt in raw_data[split]]
            tgt_sentences = [tgt for src, tgt in raw_data[split]]
            
            tokenized_src_list = []
            tokenized_tgt_list = []

            # Process in batches
            for i in tqdm(range(0, len(src_sentences), batch_size), desc=f"Tokenizing {split}", position=1, leave=False):
                batch_src = src_sentences[i:i + batch_size]
                batch_tgt = tgt_sentences[i:i + batch_size]
                tokenized_src = self.tokenizer_src(batch_src, padding=True, truncation=True, max_length=self.max_padding)["input_ids"]
                tokenized_tgt = self.tokenizer_tgt(batch_tgt, padding=True, truncation=True, max_length=self.max_padding)["input_ids"]
                tokenized_src_list.extend(tokenized_src)
                tokenized_tgt_list.extend(tokenized_tgt)
            # Convert to tensors
            tokenized_src = torch.tensor(tokenized_src_list)
            tokenized_tgt = torch.tensor(tokenized_tgt_list)
            preprocessed_dataset[split] = torch.stack([tokenized_src, tokenized_tgt], dim=1)
        return preprocessed_dataset
    
    @staticmethod
    def split_data(data, split_ratio=(0.8, 0.1, 0.1), random_seed=None):
        '''
        Splits a given dataset into train, validation and test sets as
        determined by the specified split ratio
        TODO: deprecate this if split is done in get_data
        '''
        train_size, val_size, test_size = split_ratio
        train_data, val_and_test_data = train_test_split(
            data,
            train_size=train_size,
            random_state=random_seed
        )
        val_data, test_data = train_test_split(
            val_and_test_data,
            train_size=val_size/(val_size+test_size),
            random_state=random_seed
        )
        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
        return train_data, val_data, test_data

    @staticmethod
    def limit_train_size(data, dataset_size, max_padding):
        """
        Limit the size of the training dataset and the max padding length
        Args:
            data: Dict of torch tensors for each split
            dataset_size: Int, maximum size of the training dataset
            max_padding: Int, maximum padding length
        Returns:
            data: Dict of torch tensors for each split, with limited size and max padding
        """
        import pdb; pdb.set_trace()
        original_dataset_size, _, original_max_padding = data['train'].shape
        # limit train size to dataset_size
        dataset_size = min(dataset_size, original_dataset_size)
        # limit training data to max_padding tokens
        max_padding = min(max_padding, original_max_padding)
        data['train'] = data['train'][:dataset_size, :, :max_padding]
        return data