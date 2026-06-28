import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import t, ttest_ind

class run_time:
    """ Usage:
    run_time_calc = run_time()
    run_time_calc.print_time()
    """
    def __init__(self):
        self.start_time = datetime.now()
    
    def block_end(self):
        self.end_time = datetime.now()
        self.diff = self.end_time - self.start_time
        self.days = self.diff.days
        self.hours = self.diff.seconds // 3600
        self.minutes = (self.diff.seconds % 3600) // 60
        self.seconds = self.diff.seconds % 60
        self.microseconds = self.diff.microseconds
    
    def print_time(self, n=20):
        dashes = "-" * (n)
        print(f"\n{dashes}\nStart time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTotal execution time:")
        print(f"{self.days} days, {self.hours} hours, {self.minutes} minutes, {self.seconds} seconds, and {self.microseconds} microseconds\n{dashes}")

class save_dir:
    def __init__(self, runtime, parent_dir, file_name, t=0):
        self.runtime = runtime
        self.parent_dir = parent_dir
        self.file_name = file_name
        self.checkpoint_dir = self.parent_dir + self.runtime.start_time.strftime('%Y-%m-%d-%H%M%S') + file_name + "/Checkpoints/"
        self.graph_dir = self.parent_dir + self.runtime.start_time.strftime('%Y-%m-%d-%H%M%S') + file_name
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    def print_dir(self):
        n = len(self.graph_dir)
        dashes = "-" * (n+15)
        print(f"\n{dashes}\nFiles saved at {self.graph_dir}\n{dashes}")
        return (n+15)

class plot_graphs:
    def __init__(self, graph_dir, window_size=100):
        self.graph_dir = graph_dir
        self.window_size = window_size
        self.zeros_list = list(np.zeros(self.window_size-1))

    def plot(self, metric, x, y, color="blue", mov_avg=False, mov_color="red"):
        if not mov_avg:
            plt.cla()
            plt.plot(np.array(x), np.array(y), color=color)
            plt.savefig(self.graph_dir + '/' + metric + '.png')

        else:
            plt.cla()
            metric_pd = pd.DataFrame(np.hstack((np.array(self.zeros_list), np.array(y))), columns=[metric])
            metric_avg = metric_pd.rolling(self.window_size).mean()[self.window_size-1:]
            metric_avg.reset_index(inplace=True)
            metric_avg = metric_avg[metric]
            plt.plot(np.array(x), np.array(y), color="blue")
            plt.plot(metric_avg, color=mov_color)
            plt.savefig(self.graph_dir + '/' + metric + '_mov_avg.png')

def calculate_stats(metric_data, metric="metric"):
    mean_data = np.mean(metric_data)
    std_data = np.std(metric_data)
    return "{}: {} ± {}".format(metric, np.round(mean_data, 4), np.round(std_data, 4))