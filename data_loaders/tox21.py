from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Tox
import numpy as np
from transformer.fingerprints import FingerprintsTransformer
from data_loaders.dataset import DataLoader

# https://tdcommons.ai/single_pred_tasks/tox/#tox21
class Tox21(DataLoader):
    def __init__(self):
        label_list = retrieve_label_name_list('Tox21')
        data = Tox(name = 'Tox21', label_name = label_list[0]).get_data()

        transformer = FingerprintsTransformer(data, "Drug", "ECFP")
        
        self.name = "Tox21"
        self.x_values = transformer.to_np()
        self.y_values = data["Y"].to_numpy()

    def size(self):
        return len(self.x_values)

    def x(self, dataset_slice_indices: np.ndarray) -> np.ndarray:
        return self.x_values[dataset_slice_indices]

    def y(self, dataset_slice_indices):
        return self.y_values[dataset_slice_indices]