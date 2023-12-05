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
    def __init__(self):
        self.metrics = defaultdict(list)

    def log(self, name, value):
        '''
        Appends metric to existing log of its history over training.
        '''
        self.metrics[name].append(value)

    def saveplot(self, metric_names, title, title_dict, plot_type):
        '''
        Plots and saves the metric history for specified list of metrics.
        '''

        # compute plot limits
        if plot_type == 'loss':
            ylim = (0, 8)
        elif plot_type == 'bleu':
            ylim = (0, 1)
        else: 
            raise ValueError(f"Invalid plot_type '{plot_type}'")
        
        composed_title = (" | ").join(
            [title] + 
            [f"{k.replace('_', ' ').capitalize()}: {v}" 
             for k, v in title_dict.items()]
        )
        
        plt.figure()
        for name in metric_names:
            label = name.replace('_',' ').capitalize()
            metric_history = self.metrics[name]
            plt.ylim(*ylim)
            plt.xlim(1, len(metric_history))
            plt.plot(range(1, len(metric_history)+1), metric_history, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title(composed_title)
        plt.grid(visible=True)
        plt.autoscale(False)
        plt.legend()
        # whole number ticks
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        # save plot
        save_path = (f"artifacts/loss_curves/"
                     f"N{title_dict['N']}/{title.lower().replace(' ', '_')}"
                     f"_bs_{title_dict['batch_size']}.png")
        plt.show()
        plt.pause(0.01)
        plt.savefig(save_path)

class TranslationLogger:
    def __init__(self):
        self.sentences = defaultdict(list)
        self.metrics = defaultdict(float)

    def log(self, name, value):
        '''
        Logs sentences like target sentence, predicted sentence and so on.
        '''
        self.sentences[name].append(value)
    
    def log_metric(self, name, value):
        '''
        Logs numerical metrics like BLEU score of the entire data.
        '''
        self.metrics[name] = value

    def print_and_save(self, base_path, title, title_dict):
        '''
        Prints translations and corresponding metrics and saves them as an image.
        '''
        # Compose a title for the image
        composed_title = (" | ").join(
            [title] + 
            [f"{k.replace('_', ' ').capitalize()}: {v}" 
             for k, v in title_dict.items()]
        )
        # Start composing pretty print of translated text
        print_text = f"{composed_title}\n"
        # get number of examples
        num_examples = len(list(self.sentences.values())[0])
        for i in range(num_examples):
            print_text += f'\nExample {i+1}:\n' 
            for name in self.sentences:
                print_text += f'{name}: {self.sentences[name][i]}\n'

        for name in self.metrics:
            print_text += f'\n{name}: {self.metrics[name]:.4f}'
        
        # print and save as image
        print(print_text)
        save_path = (base_path + 
                     f"N{title_dict['N']}/epoch_{title_dict['epoch']:02d}.png")
        plt.figure()
        plt.text(0, 1, print_text)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
