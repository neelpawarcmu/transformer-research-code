from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

class DirectoryCreator:
    base_path = 'artifacts'
    @classmethod
    def add_dir(cls, name, include_base_path=False):
        '''
        Recursively creates a single directory, if it does not exist.
        '''
        if include_base_path: 
            name = f'{cls.base_path}/{name}'
        os.makedirs(name, exist_ok = True)
        
    @classmethod
    def create_dirs(cls, dirs):
        '''
        Creates multiple directories from a list of directory names.
        '''
        for dir in dirs:
            cls.add_dir(dir, include_base_path=True)
            
class TrainingLogger:
    def __init__(self, N, batch_size):
        self.metrics = defaultdict(list)
        self.N = N
        self.bs = batch_size

    def log(self, name, value):
        '''
        Appends metric to existing log of its history over training.
        '''
        self.metrics[name].append(value)

    def saveplot(self, names: list, title):
        '''
        Plots and saves the metric history for specified list of metrics.
        '''
        for name in names:
            label = name.replace('_',' ').capitalize()
            plt.plot(self.metrics[name], label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title(title)
        plt.grid(visible=True)
        plt.autoscale(False)
        plt.legend()
        # whole number ticks
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        # save plot
        save_path = f"artifacts/loss_curves/N{self.N}/{title.lower()} bs_{self.batch_size}.png"
        plt.savefig(save_path)

class TranslationLogger:
    def __init__(self, N, epoch, num_examples):
        self.sentences = defaultdict(list)
        self.metrics = defaultdict(float)
        self.N = N
        self.epoch = epoch
        self.num_examples = num_examples

    def log(self, name, value):
        '''
        Logs sentences like target sentence, predicted sentence and so on.
        '''
        self.sentences[name].append(value)
    
    def log_metric(self, name, value):
        '''
        Logs numerical metrics like BLEU score of the entire data.
        '''
        self.sentences[name] = value

    def print_and_save(self, base_path):
        '''
        Prints translations and corresponding metrics and saves them as a plot.
        '''
        # Generate pretty print of translated text
        print_text = f"Transformer layers: {self.N} | Epoch: {self.epoch}\n"
        for i in range(self.num_examples):
            print_text += f'\nExample {i+1}:\n' 
            for name in self.sentences:
                print_text += f'{name}: {self.sentences[name][i]}\n'

        for name in self.metrics:
            print_text += f'\n{name}: {self.metrics[name]:.4f}'
        
        # print and save as plot
        print(print_text)
        save_path = (base_path + 
                     f"N{self.N}/epoch_{self.epoch:02d}.png")
        plt.figure()
        plt.text(0, 1, print_text)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
