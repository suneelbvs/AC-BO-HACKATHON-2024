import numpy as np
import pandas as pd
import os
import ast

from data_loaders.dataset import DataLoader
from functools import lru_cache

class Ames(DataLoader):
<<<<<<< HEAD
    fingerprint = 'Mordred'
    
=======

    fingerprint= 'Mordred'

>>>>>>> 788892f (Fix fingerprint attributes)
    def __init__(self):
        # We're using the Mordred fingerprint
        # This dataset was generated using Mordred.py
        dataset_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../datasets/fingerprint_datasets/Ames_Mordred.csv"
        )

        data = pd.read_csv(dataset_dir)
        fingerprint_arrays = data["fingerprint_Mordred"].apply(
            lambda array_str: np.array(ast.literal_eval(array_str))).tolist()
        self.x_values = np.stack(fingerprint_arrays)
        self.y_values = data["Y"].to_numpy()
        self.name = "Ames"

    def size(self):
        return len(self.x_values)
    
    @lru_cache
    def hit_threshold(self):
        return 0.5 # binary classification

    def x(self, dataset_slice: slice | np.ndarray = None) -> np.ndarray:
        if dataset_slice is None:
            return self.x_values
        return self.x_values[dataset_slice]

    def y(self, dataset_slice: slice | np.ndarray = None) -> np.ndarray:
        if dataset_slice is None:
            return self.y_values
        return self.y_values[dataset_slice]
