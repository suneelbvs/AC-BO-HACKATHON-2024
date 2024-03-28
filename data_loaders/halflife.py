import os
import numpy as np
import pandas as pd
from data_loaders.dataset import DataLoader
from transformer.fingerprints import FingerprintsTransformer

class Halflife(DataLoader):

    def __init__(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../datasets/cleaned_datasets/halflife_dataset.csv"
        )
        self.data = pd.read_csv(path)
        transformer = FingerprintsTransformer(self.data, "Drug", "ECFP")

        self.name = "halflife"
        self.x_values = transformer.to_np()
        self.y_values = self.data["Y"].to_numpy()

    def size(self):
        return len(self.x_values)

    def x(self, dataset_slice: slice | np.ndarray = None) -> np.ndarray:
        if dataset_slice is None:
            return self.x_values
        return self.x_values[dataset_slice]

    def y(self, dataset_slice: slice | np.ndarray = None) -> np.ndarray:
        if dataset_slice is None:
            return self.y_values
        return self.y_values[dataset_slice]
