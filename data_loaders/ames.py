import numpy as np
import pandas as pd
import os
import ast

from data_loaders.dataset import DataLoader

class Ames(DataLoader):
    def __init__(self):
        # We're using the Mordred fingerprint
        # This dataset was generated using Mordred.py
        dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../Fingerprints/Ames_Mordred.csv"

        data = pd.read_csv(dataset_dir)
        fingerprint_arrays = data["fingerprint_Mordred"].apply(lambda array_str: np.array(ast.literal_eval(array_str))).tolist()
        self.x_values = np.stack(fingerprint_arrays)
        self.y_values = data["Y"].to_numpy()
        self.name = "Ames"

    def size(self):
        return len(self.x_values)

    def x(self, dataset_slice_indices: np.ndarray) -> np.ndarray:
        return self.x_values[dataset_slice_indices]

    def y(self, dataset_slice_indices):
        return self.y_values[dataset_slice_indices]