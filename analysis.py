import numpy as np
import pandas as pd 
import os 
import ast


#check the tdc dataset 
from data_loaders.tox21 import Tox21
tox21 = Tox21()
data = tox21.data #this is pd dataframe
from transformer.fingerprints import FingerprintsTransformer
transformer = FingerprintsTransformer(data,"Drug",'ECFP')
data = transformer.to_np()
print(data.shape)



# #read the data 
# data = pd.read_csv(r"datasets\fingerprint_datasets\tox21.csv")
# #check the data
# data['fingerprints']=data['fingerprints'].apply(ast.literal_eval)
# finger_prints = np.stack(data['fingerprints'].values)
# print(finger_prints.shape)

