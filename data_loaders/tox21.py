from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Tox
import numpy as np
import pandas as pd

from data_loaders.dataset import DataLoader

# https://tdcommons.ai/single_pred_tasks/tox/#tox21
class Tox21(DataLoader):
    def __init__(self):
        label_list = retrieve_label_name_list('Tox21')
        data = Tox(name = 'Tox21', label_name = label_list[0]).get_data()

        # TODO: use real features
        data["num_oxygen"] = data["Drug"].str.count("O")
        self.data = data

    def full_dataset(self):
        return self.data

    def x(self, dataset_slice: pd.DataFrame) -> np.ndarray:
        return np.expand_dims(dataset_slice["num_oxygen"].to_numpy(), 0)

    def y(self, dataset_slice):
        return dataset_slice["Y"].to_numpy()