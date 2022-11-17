import os
import numpy as np
import torch


class Checkpoint:
    """
    Checkpoint class to save the best model during training
    """
    def __init__(self, filepath, metric, mode="max", initial_best=0):
        self.filepath = filepath
        self.metric = metric
        self.best = initial_best  # initial value for the monitored metric
        if mode == "min":
            self.mode = np.less
        else:
            self.mode = np.greater

    def save_weights(self, folder, fname, model):
        path = self.filepath + folder
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, f'{fname}.pt'))

    def on_epoch_end(self, metric, model, opt_model=None):
        if self.mode(metric, self.best):
            print(f"Saving new best model with {self.metric}: {round(metric, 5)}")
            self.save_weights(model[0], model[1], model[2])
            self.best = metric
            if opt_model:
                self.save_weights(opt_model[0], opt_model[1], opt_model[2])
        else:
            print("No new best model was found.")
