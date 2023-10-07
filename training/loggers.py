from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

class Logger:
    def __init__(self, N):
        self.metrics = defaultdict(list)
        self.N = N

    def log(self, name, value):
        '''
        Appends metric to existing log of metrics
        '''
        self.metrics[name].append(value)

    def plot(self, name: str, newplot=False, save=False):
        '''
        Plot and save the metric history for specified metric
        '''
        if newplot: 
            plt.figure(dpi=300)
            ylabel = f"{name.split('_')[-1]}"
        plt.plot(self.metrics[name], label=name.replace('_', ' '))
        plt.xlabel("Epoch")
        plt.ylabel(f"{ylabel.capitalize()}")
        plt.grid(visible=True)
        plt.legend()
        # set x-axis ticks as whole numbers corresponding to epochs
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        if save:
            savepath = f"artifacts/loss_curves/N{self.N}/{ylabel.lower()}.png"
            plt.savefig(savepath)