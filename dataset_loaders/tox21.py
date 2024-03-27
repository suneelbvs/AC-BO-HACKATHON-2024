from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Tox
import numpy as np

from dataset_loaders.dataset import Dataset

class Tox21(Dataset):
    def __init__(self):
        label_list = retrieve_label_name_list('Tox21')
        data = Tox(name = 'Tox21', label_name = label_list[0])
        self.data = data

    def get_x(self):
        # TODO: create real features
        self.data["num_oxygen"] = self.data["Drug"].str.count("O")
        return np.ndarray([self.data])

    def get_y(self):
        return self.data["Y"]