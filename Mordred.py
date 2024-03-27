from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
from mordred import Calculator, descriptors
from sklearn.preprocessing import MinMaxScaler

smi_example = ['CN1C=NC2=C1C(=O)N(C(=O)N2C)C','CCCc1ccncc1C(=O)NC']
mols = [Chem.MolFromSmiles(smi) for smi in smi_example]


#################################################################
Dataset = 'Ames'
#df_data = pd.read_csv(r'C:\Users\js-ne\Downloads\AMES_dataset.csv')
df_data = pd.read_csv("/home/student/j_spie17/Hackathon/AMES_dataset.csv")

Raw = False
Numeric = False
Scaler = True
Correlation = True

if Raw == True:

    mols = [Chem.MolFromSmiles(smi) for smi in df_data['Drug'].tolist()]
    calc = Calculator(descriptors, ignore_3D=False)
    df_mor = calc.pandas(mols)

    df_mor.to_csv(f'/home/student/j_spie17/Hackathon/Mordred/raw_{Dataset}', index=False)

if Numeric == True:
    df_raw = pd.read_csv(f'/home/student/j_spie17/Hackathon/Mordred/raw_{Dataset}')
    non_numeric_columns = []
    numeric_columns = []
    columns = df_raw.columns
    columns = columns.to_list()
    columns = columns[1:]

    for column in columns:
        # Try converting the column to numeric using 'pd.to_numeric'
        try:
            pd.to_numeric(df_raw[column])
            numeric_columns.append(column)
        except:
            # If 'pd.to_numeric' raises a ValueError, it means the column contains non-numeric values
            non_numeric_columns.append(column)

    df_numeric = pd.DataFrame()
    for col in numeric_columns:
        df_numeric[col] = df_raw[col].tolist()

    df_numeric.to_csv(f'/home/student/j_spie17/Hackathon/Mordred/numeric_{Dataset}', index=False)

if Scaler == True:
    scaler = MinMaxScaler()

    df_numeric = pd.read_csv(f'/home/student/j_spie17/Hackathon/Mordred/numeric_{Dataset}')
    df_scaled = pd.DataFrame()
    # Fit and transform the data using the scaler
    for col in df_numeric.columns:
        df_scaled[col] = scaler.fit_transform(df_numeric[[col]])

    df_scaled.to_csv(f'/home/student/j_spie17/Hackathon/Mordred/scaled_{Dataset}', index=None)



print('done')
