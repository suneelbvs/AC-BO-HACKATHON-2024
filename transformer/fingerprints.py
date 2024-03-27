import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from mordred import Calculator, descriptors

class FingerprintsTransformer:
    def __init__(self, dataset: pd.DataFrame, col_mol:str, fingerprint_type:str):
        self.dataset = dataset
        self.col_mol = col_mol
        #self.mols = self.prepare_molecules()
        self.fingerprint_type = fingerprint_type

    def prepare_molecules(self):
        #TODO: delete not necessary
        """
        Converts SMILES strings in the specified column to RDKit molecule objects and replaces the column.
        """
        # Convert SMILES to molecule objects and replace the column in the DataFrame
        self.dataset["SMILES"] = self.dataset[self.col_mol].apply(lambda smi: Chem.MolFromSmiles(smi) if pd.notnull(smi) else None)

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
        # Convert SMILES to molecule objects
        mols = smiles_series.apply(Chem.MolFromSmiles)
        
        # Generate ECFP fingerprints for each molecule
        ecfp_vector = mols.apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius, bits) if mol else None)
        
        # Convert fingerprints to list of integers
        ecfp_list = ecfp_vector.apply(lambda fp: list(map(int, DataStructs.BitVectToText(fp))) if fp else None)

        return ecfp_list