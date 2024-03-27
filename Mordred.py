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

Raw = True
Numeric = True
Scaler = True

if Raw == True:

    mols = [Chem.MolFromSmiles(smi) for smi in df_data['Drug'].tolist()]
    calc = Calculator(descriptors, ignore_3D=True)
    df_mor = calc.pandas(mols)

    df_mor.to_csv(f'/home/student/j_spie17/Hackathon/Mordred/raw_{Dataset}', index=false)

if Numeric == True:
    df_raw = pd.read_csv(f'/home/student/j_spie17/Hackathon/Mordred/raw_{Dataset}')
    non_numeric_columns = []
    numeric_columns = []
    columns = df_leon_mordred_raw.columns
    columns = columns.to_list()
    columns = columns[1:]

    for column in columns:
        # Try converting the column to numeric using 'pd.to_numeric'
        try:
            pd.to_numeric(df_leon_mordred_raw[column])
            numeric_columns.append(column)
        except:
            # If 'pd.to_numeric' raises a ValueError, it means the column contains non-numeric values
            non_numeric_columns.append(column)



print('done')
