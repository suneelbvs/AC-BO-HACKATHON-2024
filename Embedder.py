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

smi_example = ['CN1C=NC2=C1C(=O)N(C(=O)N2C)C','CCCc1ccncc1C(=O)NC']
mols = [Chem.MolFromSmiles(smi) for smi in smi_example]


#################################################################
Dataset = 'Ames'
#df_data = pd.read_csv(r'C:\Users\js-ne\Downloads\AMES_dataset.csv')
df_data = pd.read_csv("/home/student/j_spie17/Hackathon/AMES_dataset.csv")
#################################################################



#################################################################
#create fingerprints
fingerprint = 'Mordred'

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
    calc = Calculator(descriptors, ignore_3D=True)
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
df_fp.to_csv(f'D:\\Users\\js-ne\\PycharmProjects\\hackathon_24\\AC-BO-HACKATHON-2024\\Fingerprints\\{Dataset}_{fingerprint}.csv', index=False)

print('done')