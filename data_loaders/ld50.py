import os
import numpy as np
import pandas as pd
import ast

from data_loaders.dataset import DataLoader
from functools import lru_cache

class LD50(DataLoader):

    def __init__(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../datasets/fingerprint_datasets/LD50_Zhu_ECFP.h5"
        )
        self.data = pd.read_hdf(path)
        self.name = "ld50"
        self.fingerprint = 'ECFP'
        self.x_values = np.stack(self.data['fingerprint_ECFP'])
        self.y_values = self.data["Y"].to_numpy()

    def size(self):
        return len(self.x_values)

    @lru_cache
    def hit_threshold(self):
        sorted_y_values = np.sort(self.y_values)
        hit_threshold = np.percentile(sorted_y_values, 90)
        return hit_threshold


    def x(self, dataset_slice: slice | np.ndarray = None) -> np.ndarray:
        if dataset_slice is None:
            return self.x_values
        return self.x_values[dataset_slice]

    def y(self, dataset_slice: slice | np.ndarray = None) -> np.ndarray:
        if dataset_slice is None:
            return self.y_values
        return self.y_values[dataset_slice]