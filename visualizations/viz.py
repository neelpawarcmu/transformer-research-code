import matplotlib.pyplot as plt
import os

def visualize_attn_mask(mask):
    '''
    Assumes shape of mask to be [batch_size, seq_len, seq_len]
    '''
    mask = mask.cpu()
    print('visualizing and saving mask')
    batch_size, seq_len, seq_len = mask.shape
    os.makedirs('artifacts/plots/masks', exist_ok=True)
    for i in range(batch_size):
        mask_for_sentence = mask[i,:,:]
        plt.imshow(mask_for_sentence, interpolation='nearest')
        plt.savefig(f'artifacts/plots/masks/example_{i}')

'''
Usage:
from visualizations.viz import visualize_attn_mask
visualize_attn_mask(batch.decoder_attn_mask)
'''