import numpy as np
import pandas as pd 
import os 

#check the tdc dataset 
from data_loaders.tox21 import Tox21
tox21 = Tox21()
data = tox21.data #this is pd dataframe
from transformer.fingerprints import FingerprintsTransformer
transformer = FingerprintsTransformer(data,"Drug",'ECFP')
transformer.transform()

data_new = transformer.dataset

data_new.head()
#save
data_new.to_csv(r"transformer\tox21.csv",index=False)