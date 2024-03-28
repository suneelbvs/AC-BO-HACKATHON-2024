from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
from mordred import Calculator, descriptors
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import ast




#################################################################
Dataset = 'Ames'
fingerprint = 'Mordred'
#df_data = pd.read_csv(r'C:\Users\js-ne\Downloads\AMES_dataset.csv')
df_data = pd.read_csv("/home/student/j_spie17/Hackathon/AMES_dataset.csv")

Raw = False
Numeric = False
Scaler = False
Correlation = False
recombine = True

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
        l = scaler.fit_transform(df_numeric[[col]]).tolist()
        df_scaled[col] = [float(inner_list[0]) for inner_list in l]

    df_scaled.to_csv(f'/home/student/j_spie17/Hackathon/Mordred/scaled_{Dataset}', index=None)

if Correlation == True:
    df_scaled = pd.read_csv(f'/home/student/j_spie17/Hackathon/Mordred/scaled_{Dataset}')

    corr_matrix = df_scaled.corr()
    strong_corr_lst = []
    rows = corr_matrix.index
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if corr_matrix.iloc[j, i] > 0.8:
                strong_corr_lst.append(rows[j])
    strong_corr_lst = set(strong_corr_lst)
    strong_corr_lst = list(strong_corr_lst)

    df_no_corr = df_scaled.drop(strong_corr_lst, axis=1)
    df_no_corr.to_csv('jan_mordred_no_corr')

    df_no_corr.to_csv(f'/home/student/j_spie17/Hackathon/Mordred/no_corr_{Dataset}', index=False)

if recombine == True:
    df_no_corr = pd.read_csv(f'/home/student/j_spie17/Hackathon/Mordred/no_corr_{Dataset}')
    fp = df_no_corr.values.tolist()
    df_fp = pd.DataFrame({
        'Drug_ID': df_data['Drug_ID'].tolist(),
        'Smile': df_data['Drug'].tolist(),
        f'fingerprint_{fingerprint}': fp,
        'Y': df_data['Y'].tolist()
    })

    df_fp.to_csv(f'/home/student/j_spie17/Hackathon/Fingerprints/{Dataset}_{fingerprint}.csv', index=False)
print('done')
