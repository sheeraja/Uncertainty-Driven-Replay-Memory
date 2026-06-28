import pickle
import pprint
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
from datetime import datetime
import os


class Logger:
    def __init__(self,log_folder_details=None,train_details=None):
        self.memory = {}
        self.log_folder_details = log_folder_details
        self.train_details = train_details
        game_name = self.log_folder_details.split("-")[1]
        filedir = os.path.realpath(__file__)
        folder_idx = filedir.rfind("/")
        folder_dir = filedir[:folder_idx+1]

        today = datetime.now()
        if self.log_folder_details is None:
            directory = folder_dir + 'results/' + game_name + '/' + today.strftime('%Y-%m-%d-%H%M%S') 
        else:
            directory = folder_dir + 'results/' + game_name + '/' + today.strftime('%Y-%m-%d-%H%M%S') + '-' + self.log_folder_details
        
        os.makedirs(directory)
        self.log_folder = directory
        self.root_folder = folder_dir + 'results/' + game_name + '/'
        n = len(directory) + 15
        dashes = "-" * n
        print(f"\n{dashes}\nFiles saved at {directory}\n{dashes}")

        self.train_details['output_folder'] = directory

        with open(self.log_folder + '/' + 'experimental-setup', 'w') as handle:
                pprint.pprint(self.train_details, handle)

    def add_scalar(self, name, data, timestep):
        """
        Saves a scalar
        """
        if isinstance(data, torch.Tensor):
            data = data.item()

        self.memory.setdefault(name, []).append([data, timestep])

    def save(self):
        filename = self.log_folder + '/log_data.pkl'
        
        with open(filename, 'wb') as output:
            pickle.dump(self.memory, output, pickle.HIGHEST_PROTOCOL)

        self.save_graphs()

    def save_graphs(self):
        for key in self.memory.keys():
            plt.cla()
            plt.plot(np.array(self.memory[key])[:,1],np.array(self.memory[key])[:,0])
            if self.log_folder is None:
                plt.savefig(key+'.png')
            else:
                plt.savefig(self.log_folder + '/' + key+'.png')

