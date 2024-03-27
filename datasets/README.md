# Datasets:

# hERG Central (Source: TDC)

## Dataset Description
The Human ether-à-go-go related gene (hERG) is crucial for the coordination of the heart's beating. 
Thus, if a drug blocks the hERG channel, it could lead to severe adverse effects. 
Therefore, the reliable prediction of hERG liability in the early stages of drug design is essential to reduce the risk of cardiotoxicity-related attritions in the later development stages. 
The dataset targets three main areas: `hERG_at_1uM`, `hERG_at_10uM`, and `hERG_inhib`.

### About potency:

- ** Label : hERG_inhib (Binary classification):**
- Given a drug's SMILES string, predict whether it blocks (1) or does not block (0) the hERG channel.
- This classification is based on whether `hERG inhibition_at_10uM < -50`, i.e., whether the compound has an IC50 of less than 10µM.

## Cleaned Dataset
Upon reviewing the dataset for property distribution, it was observed that the compounds are generally larger molecules, even with molecular weights > 1000, ring counts of 10, and TPSA > 400. To ensure the dataset's relevance and manageability, property filters were applied to select compounds with the following characteristics:
- Molecular Weight (`mol_wt`) greater than 200 and less than 550
- Number of Rings (`num_rings`) less than 8
- LogP between 5 and 15

These filters were chosen to focus on compounds with properties that are more typical of drug-like molecules, thereby enhancing the relevance of the dataset for predictive modeling and drug design.

## Data Statistics
| Dataset             | Total Compounds | Number of 0s | Number of 1s |
|---------------------|-----------------|--------------|--------------|
| Original Data       | 306,893         | 293,149      | 13,744       |
| Cleaned Dataset     | 288,787         | 275,880      | 12,907       |


# AMES Mutagenicity (Source: TDC)


# Half life (Source: TDC)
The half-life of a drug refers to the time it takes for the concentration of the drug in the body to be reduced by half. 
It is a measure of the drug's duration of action. This dataset is associated with the CHEMBL assay ID 1614674.

## Task Description

- **Type:** Regression (value represents halflife in hours) 
- **Objective:** Given a drug's SMILES string, predict the duration of its half-life.

## Dataset Statistics

| Dataset Description | Total Compounds |
|---------------------|-----------------|
| Original Data       | 667             |
| Cleaned Dataset     | 489             |


