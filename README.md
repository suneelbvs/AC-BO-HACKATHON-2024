# AC-BO-HACKATHON-2024
## Project 17: Comparative Analysis of Acquisition Functions in Bayesian Optimization for Drug Discovery

### Abstract
This project conducts a comparative analysis of acquisition functions in Bayesian Optimization (BO) for drug discovery, focusing on small, diverse, unbalanced, and noisy datasets. It evaluates the impact of different acquisition functions, molecular featurization methods, and applicability domain (AD) to uncover optimal strategies for employing AF effectively in drug discovery challenges.

### References
- Hugo Bellamy (2022), "Batched Bayesian Optimization for Drug Design in Noisy Environments," J Chem Inf Model. 2022 Sep 12; 62(17): 3970–3981.

### Contributors
- Jan Christopher Spies, University of Muenster
- Jakub Lála, Imperial College London
- Yunheng Zou, University of Waterloo
- Luis Walter, Heidelberg University
- Curtis Chong, University of Waterloo

### Repository Structure
```
AC-BO-HACKATHON-2024/
│
├── src/
│   ├── <Bayesian Optimization and acquisition function scripts>
│
├── datasets/
│   ├── <Drug discovery datasets>
│
├── featurization_methods/
│   ├── fingerprints/
│   │   ├── <Scripts for molecular fingerprint generation>
│   ├── graph/
│       ├── <Graph-based molecular featurization tools>
│
├── ML_and_DL_models/
│   ├── <Machine learning and deep learning models>
│
├── metrics/
│   ├── <Performance evaluation scripts>
│
├── plots_for_BO/
│   ├── <Visualization scripts and generated plots>
│
└── applicability_domain/
    ├── <Applicability domain assessment methods>
```

# update the names for the below tasks
Task items:
| Task Item                   | Team Member          | Package/Tool       |
|-----------------------------|----------------------|--------------------|
| Dataset Selection           | Jan Christopher Spies| N/A                |
| Data Preparation            | Yunheng Zou          | RDKit, OpenBabel   |
| Data Featurization          |                      |                    |
| - Fingerprints              | Jakub Lála           | RDKit              |
| - Graph                     | Luis Walter          | DeepChem           |
| ML Models                   | Curtis Chong         | Scikit-learn       |
| DL Models                   | Jakub Lála           | PyTorch            |
| Metrics                     | Yunheng Zou          | Scikit-learn       |
| Applicability Domain (AOD)  | Luis Walter          | Scikit-learn       |
| Plotting                    | Jan Christopher Spies| Matplotlib         |

### Installation
To install an editable version of the package, run the following command:
```bash
pip install PyTDC==0.3.6
pip install xgboost
pip install torch
pip install gpytorch
pip install requests
```
