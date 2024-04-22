import os
import mlflow
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.interpolate as interp
from torch.utils.tensorboard import SummaryWriter

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

class BaseLogger:
    def __init__(self, config):
        self.config = config
        self.tracking_uri = "http://127.0.0.1:8080"
        self.experiment_name = config["experiment_name"]
        run_name = "_".join([f'{k}_{config[k]}' for k in ["N", 
                                                         "dataset_size",
                                                         "random_seed",
                                                         ]])
        self.run_name = run_name
        self.client = mlflow.tracking.MlflowClient()
        self.writer = SummaryWriter()
        mlflow.start_run()
        mlflow.set_experiment(self.experiment_name)

    # def autolog(self):
    #     mlflow.autolog()
    
    def upload_artifacts(self):
        raise NotImplementedError


class TrainingLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        self.metrics = defaultdict(list)

    def log(self, name, value, step):
        '''
        Appends metric to existing log of its history over training.
        '''
        self.metrics[name].append(value)
        mlflow.log_metric(name, value)
        name_for_tensorboard = "/".join(name.split("_")[::-1])
        self.writer.add_scalar(name_for_tensorboard, value, step)

    def saveplot(self, epoch_num, metric_names, title, title_dict, plot_type, xlabel="Epoch"):
        '''
        Plots and saves the metric history for specified list of metrics.
        '''

        # compute plot limits
        if plot_type == 'loss':
            ylim = (0, 9)
        elif plot_type == 'bleu':
            ylim = (0, 1)
        else: 
            raise ValueError(f"Invalid plot_type '{plot_type}'")
        
        composed_title = (" | ").join(
            [title] + 
            [f"{k.replace('_', ' ').capitalize()}: {v}" 
             for k, v in title_dict.items()]
        )

        max_length = max(
            [len(history) for _, history in self.metrics.items()]
        )
    
        fig, ax = plt.subplots()
        for name in metric_names:
            label = name.replace('_',' ').capitalize()
            metric_history = self.interpolate(self.metrics[name], max_length)
            ax.plot(range(1, len(metric_history)+1), metric_history, label=label)
        ax.set_ylim(ylim)
        ax.set_xlim(1, len(metric_history))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(plot_type.capitalize())
        ax.set_title(composed_title, y=1.08)

        # plot secondary axis of epochs
        ax2 = ax.twiny()
        ax2.set_xlim(1, epoch_num)
        ax2.set_xlabel("Epoch")
        # ax2.set_xticks(range(1, epoch_num + 1))
        ax.grid(visible=True)
        ax.legend()

        # save plot
        save_path = (f"artifacts/loss_curves/"
                    f"N{title_dict['N']}/{title.lower().replace(' ', '_')}"
                    f"_dataset_size_{title_dict['dataset_size']}.png")
        plt.show()
        plt.pause(0.01)
        fig.savefig(save_path)
        plt.close()
        mlflow.log_artifacts(f"artifacts/loss_curves/N{self.config['N']}")
        self.writer.flush()

    def interpolate(self, array, target_length):
        
        mesh = interp.interp1d(np.arange(len(array)), array)
        interpolated_array = mesh(np.linspace(0,len(array)-1,target_length)).tolist()
        return interpolated_array

    def close(self):
        mlflow.end_run()
        self.writer.close()

class AutoTrainingLogger(TrainingLogger):
    def __init__(self, config):
        super().__init__(config)

class TranslationLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        self.sentences = defaultdict(list)
        self.metrics = defaultdict(float)

    def log_sentence(self, name, value):
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
