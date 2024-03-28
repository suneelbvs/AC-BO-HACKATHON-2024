'''
Task:
-load Datasets
-create feature vectors
-exportr features in a .csv
'''

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
from mordred import Calculator, descriptors
import h5py




#################################################################
Dataset = 'Ames'

if Dataset == 'Ames':
    df_data = pd.read_csv("/home/student/j_spie17/Hackathon/AMES_dataset.csv")

elif Dataset == 'halflife':
    df_data = pd.read_csv("/home/student/j_spie17/Hackathon/halflife_dataset.csv")
    
elif Dataset == 'HERG':
    df_data = pd.read_csv("HERG_dataset_final.csv")
#################################################################



#################################################################
#create fingerprints
fingerprint = 'ECFP'

def ECFP(mols,radius=2,bits=2048):
    ECFPS_vector = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in mols]
    ECFPS_list = [list(map(int, DataStructs.BitVectToText(fp))) for fp in ECFPS_vector]
    return ECFPS_list

def Mordred(mols):
    calc = Calculator(descriptors, ignore_3D=True)
    df_mordred = calc.pandas(mols)
    return df_mordred



##############################################################
mols = [Chem.MolFromSmiles(smi) for smi in df_data['Drug'].tolist()]

if fingerprint == 'Mordred':
    calc = Calculator(descriptors, ignore_3D=False)
    df_mor = calc.pandas(mols)



if fingerprint == 'ECFP':
    fp = ECFP(mols)

df_fp = pd.DataFrame({
    'Drug_ID': df_data['Drug_ID'].tolist(),
    'Smile': df_data['Drug'].tolist(),
    f'fingerprint_{fingerprint}': fp,
    'Y': df_data['Y'].tolist()
})



#############################################################
#Export Data
df_fp.to_csv(f'/home/student/j_spie17/Hackathon/Fingerprints/{Dataset}_{fingerprint}.csv', index=False)

#export the fingerprints in bit in hdf5 format
df_fp.to_hdf(f'{Dataset}_{fingerprint}.h5', key='df_mordred', mode='w')

print('done')
