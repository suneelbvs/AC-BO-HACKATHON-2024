import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

class FingerprintsTransformer:
    def __init__(self, dataset: pd.DataFrame, col_mol:str, fingerprint_type:str):
        self.dataset = dataset
        self.col_mol = col_mol
        self.fingerprint_type = fingerprint_type
        self.transform()

    def transform(self):
        """
        Adds a new column to the DataFrame with ECFP fingerprints.
        """
        if self.fingerprint_type == 'ECFP':
            self.dataset['fingerprints'] = self.ecfp(self.dataset[self.col_mol])
        # Additional fingerprint types can be handled here as well.
        else:
            raise ValueError("Unsupported fingerprint type. Choose from 'ECFP'.")

    @staticmethod
    def ecfp(smiles_series, radius=2, bits=2048):
        """
        Generates ECFP fingerprints directly from a pandas Series of SMILES strings.
        
        :param smiles_series: pandas Series containing SMILES strings.
        :param radius: The radius of the ECFP fingerprints.
        :param bits: The size of the bit vector for the ECFP fingerprints.
        :return: pandas Series where each element is an ECFP fingerprint list.
        """
        def to_fingerprint(mol):
            """Helper function to convert a molecule to a fingerprint."""
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
                return list(map(int, DataStructs.BitVectToText(fp)))
            return None

        # Convert SMILES to molecule objects
        mols = smiles_series.apply(Chem.MolFromSmiles)
        
        # Generate ECFP fingerprints for each molecule and convert to list of integers
        ecfp_list = mols.apply(to_fingerprint)

        return ecfp_list
    
    def to_np(self):
        """
        Converts the fingerprints column to a numpy array.
        """
        return np.array(self.dataset['fingerprints'].to_list())